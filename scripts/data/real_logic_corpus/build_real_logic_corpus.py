#!/usr/bin/env python
"""Build a paired formal-logic / natural-language reasoning corpus from public
deductive-reasoning datasets with machine-derivable proofs.

Sources:
  * ProofWriter OWA (Allen AI, CC BY 4.0): gold proofs with intermediates
    (train+dev of depth-0/1/2/3/5; test splits reserved for eval/dedup).
  * PrOntoQA (Saparov & He, Apache-2.0 generator): freshly generated proofs
    (chain_of_thought) over the controlled grammar. The published
    PrOntoQA-OOD eval files are deliberately NOT ingested (eval hygiene).
  * PARARULE-Plus (Strong-AI-Lab): closed-world rule-base QA without gold
    proofs; we re-derive proofs by explicit-literal forward chaining and keep
    only questions provable WITHOUT negation-as-failure.

Every emitted example is a PAIR of documents over the modality-neutral
midtraining envelope used by synthetic-RLVL
(`{problem}\n\nSolution:\n...\nFinal answer: {answer}`):
  formal_doc: premises + derivation in the repo's LogicEngine grammar
              (`N. Formula ; RULE,refs`, rules R / ∧I / ∧E / AE(∀E) / ->E)
  nl_doc:     a deterministic 1:1 natural-language rendering: same premise
              numbering, one sentence per proof line, same line numbers,
              same conclusion and final answer.

Soundness policy (fail closed):
  * every formal trace is checked with the repo LogicEngine (open-world
    natural deduction); invalid or unconvertible examples are dropped and the
    reason is recorded.
  * closed-world negation-as-failure is NOT emulated: a negative body literal
    must be satisfied by an explicitly stated/derived negative fact, otherwise
    the example is dropped (reason `naf_unsupported` / `not_derivable_explicit`).

Grammar notes (extensions relative to the synthetic generator):
  * `AE` (parsed by the engine as ∀E) instantiates a universally quantified
    rule `Ax(Body -> Head)` to a constant before `->E`. The synthetic
    generator uses ground rules only; real rule-bases are quantified.
  * `∧E` is used for PrOntoQA AndElim examples.
  * negated literals `~P(c)` appear as premises, rule-body conjuncts and rule
    heads; the engine natively supports them (no NAF semantics implied).

Usage:
  python build_real_logic_corpus.py build --source proofwriter --raw-root RAW --out-root OUT
  python build_real_logic_corpus.py build --source prontoqa   --raw-root RAW --out-root OUT
  python build_real_logic_corpus.py build --source pararule   --raw-root RAW --out-root OUT
  python build_real_logic_corpus.py dedup --raw-root RAW --out-root OUT
  python build_real_logic_corpus.py audit --out-root OUT [--tokenizer tokenizer.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_cands = [SCRIPT_DIR] + list(SCRIPT_DIR.parents)[:3]
for _cand in _cands:
    if (_cand / "logic_engine").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from logic_engine import LogicEngine  # noqa: E402

ENGINE = LogicEngine()


class Drop(Exception):
    """Raised by converters when an example cannot be soundly converted."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------
# Literals and naming
# --------------------------------------------------------------------------

RESERVED_SINGLE = set("stuvwxyz")  # single letters the engine parses as variables


def norm_const(raw: str) -> str:
    tok = re.sub(r"[^A-Za-z0-9]+", "_", raw.strip()).strip("_").lower()
    if not tok or not tok[0].isalpha():
        tok = "c_" + tok
    if len(tok) == 1 and tok in RESERVED_SINGLE:
        tok = tok + "0"
    return tok


def norm_pred(raw: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", raw.strip())
    name = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not name:
        raise Drop("bad_predicate", raw)
    if not name[0].isalpha():
        name = "P" + name
    return name


@dataclass(frozen=True)
class Lit:
    pred: str
    args: tuple  # constants (str) or the variable marker "?x"
    neg: bool = False

    def subst(self, entity: str) -> "Lit":
        return Lit(self.pred, tuple(entity if a == "?x" else a for a in self.args), self.neg)

    @property
    def ground(self) -> bool:
        return "?x" not in self.args

    def formula(self, var: str = "x") -> str:
        args = ",".join(var if a == "?x" else a for a in self.args)
        atom = f"{self.pred}({args})"
        return f"~{atom}" if self.neg else atom

    def complement(self) -> "Lit":
        return Lit(self.pred, self.args, not self.neg)


def conj_formula(lits) -> str:
    return " & ".join(l.formula() for l in lits)


def _mk_heads(head) -> tuple:
    return tuple(head) if isinstance(head, (list, tuple)) else (head,)


@dataclass
class RuleP:
    """A (possibly quantified) Horn-style rule premise; the head may be a
    conjunction of literals (PrOntoQA AndElim grammar)."""

    body: list  # list[Lit], possibly containing "?x"
    head: object  # Lit or list[Lit]
    quantified: bool

    def __post_init__(self):
        self.heads = _mk_heads(self.head)
        self.head = self.heads[0]

    def head_formula(self, heads=None) -> str:
        return " & ".join(l.formula() for l in (heads or self.heads))

    def formula(self) -> str:
        inner = f"{' & '.join(l.formula() for l in self.body)} -> {self.head_formula()}"
        return f"Ax({inner})" if self.quantified else inner

    def grounded(self, entity: str):
        body = [l.subst(entity) for l in self.body]
        heads = [l.subst(entity) for l in self.heads]
        impl = f"{' & '.join(l.formula() for l in body)} -> {self.head_formula(heads)}"
        return body, heads, impl


# --------------------------------------------------------------------------
# Proof builder / document rendering
# --------------------------------------------------------------------------


@dataclass
class Premise:
    formula: str
    nl: str


@dataclass
class Doc:
    source: str
    id: str
    problem: str
    constants: list
    predicates: list
    premises: list  # list[Premise]
    proof_formal: list = field(default_factory=list)  # "N. formula ; just"
    proof_nl: list = field(default_factory=list)  # "N. sentence"
    conclusion_formula: str = ""
    conclusion_nl: str = ""
    answer: str = ""
    depth: int = 0
    meta: dict = field(default_factory=dict)


class ProofBuilder:
    def __init__(self, premises):
        self.premises = premises
        self.lines = []  # (n, formula, just, nl)
        self.line_of = {}

    def next_line(self) -> int:
        return len(self.premises) + len(self.lines) + 1

    def add(self, formula: str, just: str, nl: str) -> int:
        n = self.next_line()
        self.lines.append((n, formula, just, nl))
        self.line_of.setdefault(formula, n)
        return n

    def have(self, formula: str):
        return self.line_of.get(formula)


ENVELOPE_FORMAL = (
    "{problem}\n\nSolution:\nContext:\nConstants:\n{constants}\n\n"
    "Predicates:\n{predicates}\n\nPremises:\n{premises}\n\n"
    "Derivation:\n{derivation}\n\nConclusion:\n{conclusion}\n\nFinal answer: {answer}"
)
ENVELOPE_NL = (
    "{problem}\n\nSolution:\nContext:\nPremises:\n{premises}\n\n"
    "Derivation:\n{derivation}\n\nConclusion:\n{conclusion}\n\nFinal answer: {answer}"
)


def render_pair(doc: Doc):
    formal = ENVELOPE_FORMAL.format(
        problem=doc.problem,
        constants="\n".join(doc.constants),
        predicates="\n".join(doc.predicates),
        premises="\n".join(f"{i}. {p.formula}" for i, p in enumerate(doc.premises, 1)),
        derivation="\n".join(doc.proof_formal),
        conclusion=doc.conclusion_formula,
        answer=doc.answer,
    )
    nl = ENVELOPE_NL.format(
        problem=doc.problem,
        premises="\n".join(f"{i}. {p.nl}" for i, p in enumerate(doc.premises, 1)),
        derivation="\n".join(doc.proof_nl),
        conclusion=doc.conclusion_nl,
        answer=doc.answer,
    )
    return formal, nl


def finalize(doc: Doc, builder: ProofBuilder, conclusion_lit_formula: str):
    if not builder.lines:
        raise Drop("empty_proof")
    doc.proof_formal = [f"{n}. {f} ; {j}" for (n, f, j, _) in builder.lines]
    doc.proof_nl = [f"{n}. {s}" for (n, _, _, s) in builder.lines]
    if builder.lines[-1][1] != conclusion_lit_formula:
        raise Drop("conclusion_not_last", f"{builder.lines[-1][1]} != {conclusion_lit_formula}")
    doc.conclusion_formula = conclusion_lit_formula


def validate(doc: Doc):
    premises = "\n".join(f"{i}. {p.formula}" for i, p in enumerate(doc.premises, 1))
    proof = "\n".join(doc.proof_formal)
    report = ENGINE.analyze_proof(premises=premises, conclusion=doc.conclusion_formula, proof=proof)
    if not report.ok:
        bad = [f"{l.line_number}: {l.error or l.syntax_error}" for l in report.lines if not l.valid]
        raise Drop("engine_invalid", (report.error or "") + " | " + " ; ".join(bad[:3]))
    if len(doc.proof_formal) != len(doc.proof_nl):
        raise Drop("pairing_mismatch")


# --------------------------------------------------------------------------
# Shared derivation emission
# --------------------------------------------------------------------------

DISPLAY_ENTITY = {}


def disp_entity(const: str) -> str:
    return DISPLAY_ENTITY.get(const, const)


def clause(sentence: str) -> str:
    s = sentence.strip()
    if s.endswith("."):
        s = s[:-1]
    return s


def clause_join(sentences) -> str:
    return " and ".join(clause(s) for s in sentences)


def emit_rule_application(builder, rule_premise_idx, rule: RuleP, entity, body_lines, disp):
    """body_lines: list of (line_no, Lit) already available, in rule-body order."""
    g_body, g_heads, g_impl = rule.grounded(entity)
    if len(g_heads) != 1:
        raise Drop("multi_head_unsupported")
    g_head = g_heads[0]
    if len(g_body) == 1:
        body_ref = body_lines[0][0]
    else:
        acc_f = g_body[0].formula()
        acc_ref = body_lines[0][0]
        for k in range(1, len(g_body)):
            acc_f = f"{acc_f} & {g_body[k].formula()}"
            existing = builder.have(acc_f)
            if existing is None:
                nl = "Combining: " + clause_join(disp(l) for l in g_body[: k + 1]) + "."
                acc_ref = builder.add(acc_f, f"∧I,{acc_ref},{body_lines[k][0]}", nl)
            else:
                acc_ref = existing
        body_ref = acc_ref
    if rule.quantified:
        impl_line = builder.have(g_impl)
        if impl_line is None:
            ent_disp = disp_entity(entity)
            nl = (
                f"Instantiating rule {rule_premise_idx} for {ent_disp}: "
                f"if {clause_join(disp(l) for l in g_body)}, then {clause(disp(g_head))}."
            )
            impl_line = builder.add(g_impl, f"AE,{rule_premise_idx}", nl)
    else:
        impl_line = rule_premise_idx
    head_f = g_head.formula()
    existing = builder.have(head_f)
    if existing is not None:
        return existing, g_head
    nl = f"Therefore, {clause(disp(g_head))}."
    line = builder.add(head_f, f"->E,{impl_line},{body_ref}", nl)
    return line, g_head


# --------------------------------------------------------------------------
# ProofWriter converter
# --------------------------------------------------------------------------

SEXPR_TOKEN = re.compile(r'"[^"]*"|[()]|[^\s()"]+')


def parse_sexpr(s: str):
    toks = SEXPR_TOKEN.findall(s)
    pos = 0

    def rd():
        nonlocal pos
        tok = toks[pos]
        pos += 1
        if tok == "(":
            out = []
            while toks[pos] != ")":
                out.append(rd())
            pos += 1
            return out
        if tok.startswith('"'):
            return tok[1:-1]
        return tok

    out = rd()
    if pos != len(toks):
        raise Drop("sexpr_trailing", s)
    return out


PW_VARS = {"someone", "something"}


class PWTheory:
    def __init__(self, record):
        self.rec = record
        self.triples = {}
        self.rules = {}
        self.premises = []
        self.lit_text = {}
        self.entity_display = {}
        self.pred_gloss = {}
        order = []
        for name, t in record["triples"].items():
            lit = self._triple_lit(parse_sexpr(t["representation"]))
            self.triples[name] = lit
            self.lit_text[lit] = t["text"]
            order.append((lit.formula(), t["text"], ("triple", name)))
        for name, r in record["rules"].items():
            rule = self._rule(parse_sexpr(r["representation"]))
            self.rules[name] = rule
            order.append((rule.formula(), r["text"], ("rule", name)))
        self.premise_idx = {}
        for i, (formula, text, key) in enumerate(order, 1):
            self.premises.append(Premise(formula, text))
            self.premise_idx[key] = i
        self.premise_formula_idx = {}
        for i, p in enumerate(self.premises, 1):
            self.premise_formula_idx.setdefault(p.formula, i)

    def _term(self, raw):
        if raw.lower() in PW_VARS:
            return "?x"
        const = norm_const(raw)
        self.entity_display.setdefault(const, raw if raw[:1].isupper() else f"the {raw}")
        return const

    def _triple_lit(self, sx, allow_naf=False) -> Lit:
        if not (isinstance(sx, list) and len(sx) == 4):
            raise Drop("bad_triple", str(sx))
        subj, verb, obj, pol = sx
        if pol not in ("+", "-", "~"):
            raise Drop("bad_polarity", str(pol))
        if pol == "~" and not allow_naf:
            raise Drop("naf_unsupported", str(sx))
        if verb == "is":
            pred = norm_pred(obj)
            args = (self._term(subj),)
            self.pred_gloss.setdefault(pred, f"{pred}(x) = x is {obj}")
        else:
            pred = norm_pred(verb)
            args = (self._term(subj), self._term(obj))
            self.pred_gloss.setdefault(pred, f"{pred}(x,y) = x {verb} y")
        neg = pol in ("-", "~")
        return Lit(pred, args, neg)

    def _rule(self, sx) -> RuleP:
        if not (isinstance(sx, list) and len(sx) == 3 and sx[1] == "->"):
            raise Drop("bad_rule", str(sx))
        body = [self._triple_lit(b, allow_naf=True) for b in sx[0]]
        head = self._triple_lit(sx[2], allow_naf=True)
        quantified = any(not l.ground for l in body + [head])
        return RuleP(body, head, quantified)

    def display(self, lit: Lit) -> str:
        if lit in self.lit_text:
            return self.lit_text[lit]
        comp = self.lit_text.get(lit.complement())
        if comp is not None and " is " in comp:
            if lit.neg:
                return comp.replace(" is ", " is not ", 1)
            return comp.replace(" is not ", " is ", 1)
        subj = disp_entity(lit.args[0])
        subj = subj[:1].upper() + subj[1:]
        if len(lit.args) == 1:
            adj = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", lit.pred).lower()
            return f"{subj} is {'not ' if lit.neg else ''}{adj}."
        verb = lit.pred.lower()
        obj = disp_entity(lit.args[1])
        return f"{subj} {'does not ' + verb.rstrip('s') if lit.neg else verb} {obj}."


def parse_pw_proof(repr_str: str):
    """Parse a proofsWithIntermediates `representation` string into a tree.

    Grammar (observed):
      NODE   := leaf | '(' NODE+ '->' TARGET ')' | '(' NODE ')'
      TARGET := ruleK | '(' ruleK '%' intJ ')'
      leaf   := tripleK | intK | NAF... | FAIL
    """
    toks = re.findall(r"[()]|->|%|[^\s()%]+", repr_str.strip())
    pos = 0

    def rd():
        nonlocal pos
        tok = toks[pos]
        pos += 1
        if tok != "(":
            return ("leaf", tok)
        items = []
        while toks[pos] != ")":
            if toks[pos] == "->":
                pos += 1
                target = rd()
                if toks[pos] != ")":
                    raise Drop("proof_parse", repr_str)
                pos += 1
                return ("apply", items, target)
            items.append(rd())
        pos += 1
        if len(items) == 1:
            return items[0]
        return ("group", items)

    tree = rd()
    if pos != len(toks):
        raise Drop("proof_parse_trailing", repr_str)
    return tree


class _NoBind:
    pass


NO_BIND = _NoBind()


def unify_pw(body_lit: Lit, child: Lit, binding):
    """Match a rule-body literal against a child literal. `binding` is the
    current entity binding for the rule variable (NO_BIND if none yet).
    Returns the (possibly updated) binding on success, or None on failure.
    A NAF ('~') body literal only matches an explicit negative child."""
    if body_lit.pred != child.pred or body_lit.neg != child.neg or len(body_lit.args) != len(child.args):
        return None
    b = binding
    for ba, ca in zip(body_lit.args, child.args):
        if ba == "?x":
            if b is NO_BIND:
                b = ca
            elif b != ca:
                return None
        elif ba != ca:
            return None
    return b


def convert_proofwriter_question(theory: PWTheory, qid, q):
    global DISPLAY_ENTITY
    DISPLAY_ENTITY = theory.entity_display
    strategy = q.get("strategy")
    answer = q.get("answer")
    if strategy not in ("proof", "inv-proof"):
        raise Drop(f"skip_strategy_{strategy}")
    pwis = q.get("proofsWithIntermediates") or []
    if not pwis:
        raise Drop("no_proof")
    pwi = pwis[0]
    ints = {}
    for name, v in (pwi.get("intermediates") or {}).items():
        lit = theory._triple_lit(parse_sexpr(v["representation"]))
        ints[name] = lit
        theory.lit_text.setdefault(lit, v["text"])
    q_lit = theory._triple_lit(parse_sexpr(q["representation"]))

    builder = ProofBuilder(theory.premises)

    def restate_triple(name):
        lit = theory.triples.get(name)
        if lit is None:
            raise Drop("unknown_triple", name)
        f = lit.formula()
        existing = builder.have(f)
        if existing:
            return existing, lit
        idx = theory.premise_idx[("triple", name)]
        line = builder.add(f, f"R,{idx}", f"We are given: {theory.lit_text[lit]}")
        return line, lit

    def rule_and_int(target):
        if target[0] == "leaf":
            return target[1]
        if target[0] == "group":
            names = [t[1] for t in target[1] if t[0] == "leaf"]
            for n in names:
                if n.startswith("rule"):
                    return n
        raise Drop("proof_parse_target", str(target))

    def eval_node(node):
        kind = node[0]
        if kind == "leaf":
            name = node[1]
            if name.startswith("triple"):
                return restate_triple(name)
            if name.startswith("int") and name in ints:
                lit = ints[name]
                existing = builder.have(lit.formula())
                if existing:
                    return existing, lit
                raise Drop("dangling_int", name)
            if "NAF" in name or name == "FAIL":
                raise Drop("naf_unsupported", name)
            raise Drop("unknown_leaf", name)
        if kind == "group":
            raise Drop("proof_parse_group", str(node))
        _, children, target = node
        # bodies like `((triple2 triple3) -> ...)` parse the two bare leaves
        # as a single "group" child; flatten it.
        children = [c for item in children for c in (item[1] if item[0] == "group" else [item])]
        rule_name = rule_and_int(target)
        rule = theory.rules.get(rule_name)
        if rule is None:
            raise Drop("unknown_rule", rule_name)
        child_results = [eval_node(c) for c in children]
        binding = NO_BIND
        used = [False] * len(child_results)
        ordered = []
        for b in rule.body:
            matched = False
            for i, (ln, lit) in enumerate(child_results):
                if used[i]:
                    continue
                bnd = unify_pw(b, lit, binding)
                if bnd is not None:
                    binding = bnd
                    used[i] = True
                    ordered.append((ln, lit))
                    matched = True
                    break
            if not matched:
                # body literal without a matching child (NAF-satisfied in the
                # source data): accept only if explicitly stated as a premise
                # or already derived.
                cand = None
                if b.ground:
                    cand = b
                elif binding is not NO_BIND:
                    cand = b.subst(binding)
                if cand is not None:
                    ln = builder.have(cand.formula())
                    if ln is None and cand.formula() in theory.premise_formula_idx:
                        idx = theory.premise_formula_idx[cand.formula()]
                        ln = builder.add(
                            cand.formula(), f"R,{idx}", f"We are given: {theory.display(cand)}"
                        )
                    if ln is not None:
                        ordered.append((ln, cand))
                        matched = True
            if not matched:
                raise Drop("naf_unsupported" if b.neg else "rule_body_unmatched", str(b))
        if rule.quantified and binding is NO_BIND:
            raise Drop("no_binding")
        entity = binding if binding is not NO_BIND else "_"
        line, lit = emit_rule_application(
            builder, theory.premise_idx[("rule", rule_name)], rule, entity, ordered, theory.display
        )
        return line, lit

    root_line, root_lit = eval_node(parse_pw_proof(pwi["representation"]))
    if answer is True:
        if root_lit != q_lit:
            raise Drop("conclusion_mismatch", f"{root_lit} vs {q_lit}")
        ans = "True"
    elif answer is False:
        if root_lit != q_lit.complement():
            raise Drop("conclusion_mismatch_inv", f"{root_lit} vs {q_lit}")
        ans = "False"
    else:
        raise Drop("skip_unknown_answer")

    rec = theory.rec
    doc = Doc(
        source="proofwriter",
        id=f"{rec['id']}::{qid}",
        problem=f"{rec['theory']}\n\nTrue or false: {q['question']}",
        constants=sorted(f"{c} = {d}" for c, d in theory.entity_display.items()),
        predicates=sorted(theory.pred_gloss.values()),
        premises=theory.premises,
        answer=ans,
        depth=int(q.get("QDep") or 0),
        meta={"strategy": strategy, "theory_id": rec["id"]},
    )
    finalize(doc, builder, root_lit.formula())
    doc.conclusion_nl = theory.display(root_lit)
    validate(doc)
    return doc


def build_proofwriter(raw_root: Path, out, limit=None):
    base = raw_root / "proofwriter" / "proofwriter-dataset-V2020.12.3" / "OWA"
    reasons = collections.Counter()
    kept = 0
    total = 0
    for depth_dir in ["depth-0", "depth-1", "depth-2", "depth-3", "depth-5"]:
        for split in ["train", "dev"]:
            path = base / depth_dir / f"meta-{split}.jsonl"
            with open(path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    try:
                        theory = PWTheory(rec)
                    except (Drop, Exception) as d:  # noqa: BLE001
                        r = d.reason if isinstance(d, Drop) else f"error_{type(d).__name__}"
                        for _qid in rec.get("questions", {}):
                            total += 1
                            reasons[f"theory_{r}"] += 1
                        continue
                    for qid, q in rec.get("questions", {}).items():
                        total += 1
                        if limit and kept >= limit:
                            return kept, total, reasons
                        try:
                            doc = convert_proofwriter_question(theory, qid, q)
                            doc.meta["pw_depth_dir"] = depth_dir
                            doc.meta["pw_split"] = split
                            emit(out, doc)
                            kept += 1
                        except Drop as d:
                            reasons[d.reason] += 1
                        except Exception as e:  # noqa: BLE001
                            reasons[f"error_{type(e).__name__}"] += 1
    return kept, total, reasons


# --------------------------------------------------------------------------
# PrOntoQA converter
# --------------------------------------------------------------------------

ART = r"(?:a|an)"


class PQParser:
    def __init__(self, sentences):
        self.vocab = set()
        self.phrase = {}  # pred -> display phrase ('a wumpus' | 'opaque')
        for s in sentences:
            for m in re.finditer(rf"\bis {ART} ([a-z][a-z-]*)\b", s):
                self.vocab.add(m.group(1))
            for m in re.finditer(r"\b(?:Every|Each) ([a-z][a-z-]*)\b", s):
                self.vocab.add(m.group(1))

    def sing(self, w):
        w = w.lower()
        if w in self.vocab:
            return w
        if w.endswith("es") and w[:-2] in self.vocab:
            return w[:-2]
        if w.endswith("s") and w[:-1] in self.vocab:
            return w[:-1]
        if w.endswith("puses"):
            return w[:-2]
        if w.endswith("pus"):
            return w  # fictional '-pus' nouns are already singular
        if w.endswith("uses") or w.endswith("ses") or w.endswith("xes") or w.endswith("ches"):
            return w[:-2]
        if w.endswith("s"):
            return w[:-1]
        return w

    def pred_of(self, phrase, gloss):
        """phrase like 'a wumpus' | 'not a wumpus' | 'opaque' | 'not opaque'"""
        p = phrase.strip()
        neg = False
        if p.startswith("not "):
            neg = True
            p = p[4:].strip()
        m = re.fullmatch(rf"{ART} ([a-z][a-z-]*)", p)
        if m:
            noun = self.sing(m.group(1))
            pred = norm_pred(noun)
            gloss.setdefault(pred, f"{pred}(x) = x is a {noun}")
            self.phrase.setdefault(pred, f"a {noun}")
            return pred, neg
        if re.fullmatch(r"[a-z][a-z-]*", p):
            pred = norm_pred(p)
            gloss.setdefault(pred, f"{pred}(x) = x is {p}")
            self.phrase.setdefault(pred, p)
            return pred, neg
        raise Drop("pq_bad_phrase", phrase)

    def lits_of(self, phrase, entity, gloss):
        """Like pred_of but also supports 'a fruity dumpus' (adjective + noun
        conjunction, PrOntoQA AndIntro/AndElim grammar). Returns [Lit,...]."""
        p = phrase.strip()
        m = re.fullmatch(rf"{ART} ([a-z][a-z-]*) ([a-z][a-z-]*)", p)
        if m:
            adj_p, _ = self.pred_of(m.group(1), gloss)
            noun_p, _ = self.pred_of(f"a {m.group(2)}", gloss)
            return [Lit(adj_p, (entity,), False), Lit(noun_p, (entity,), False)]
        pred, neg = self.pred_of(p, gloss)
        return [Lit(pred, (entity,), neg)]


def _pq_head_pred(pq, rest, gloss, s):
    """One head conjunct of a rule: plural noun, article+noun(s), or adjective."""
    neg = False
    if rest.startswith("not "):
        neg = True
        rest = rest[4:]
    mm = re.fullmatch(rf"{ART} ([a-z][a-z-]*)", rest)
    if mm:
        return _pq_noun_pred(pq, mm.group(1), gloss), neg
    if re.fullmatch(r"[a-z][a-z-]*(es|s)", rest) and (
        pq.sing(rest) in pq.vocab or rest.endswith("uses") or rest.endswith("puses")
    ):
        return _pq_noun_pred(pq, rest, gloss), neg
    if re.fullmatch(r"[a-z][a-z-]*", rest):
        pred = norm_pred(rest)
        gloss.setdefault(pred, f"{pred}(x) = x is {rest}")
        pq.phrase.setdefault(pred, rest)
        return pred, neg
    raise Drop("pq_bad_rule", s)


def _pq_head_lits(pq, rest, gloss, s):
    """Full rule head, possibly a conjunction: 'a mean dumpus',
    'a rompus and a vumpus', 'grimpuses and impuses', 'metallic'."""
    out = []
    for part in re.split(r" and ", rest.strip()):
        part = part.strip()
        mm = re.fullmatch(rf"{ART} ([a-z][a-z-]*) ([a-z][a-z-]*)", part) or re.fullmatch(
            r"(?!an? )([a-z][a-z-]*) ([a-z][a-z-]*(?:puses|xes|ches|shes))", part
        )
        if mm:
            adj_p, _ = pq.pred_of(mm.group(1), gloss)
            noun_p = _pq_noun_pred(pq, mm.group(2), gloss)
            out += [Lit(adj_p, ("?x",), False), Lit(noun_p, ("?x",), False)]
            continue
        pred, neg = _pq_head_pred(pq, part, gloss, s)
        out.append(Lit(pred, ("?x",), neg))
    return out


def _pq_noun_pred(pq, word, gloss):
    noun = pq.sing(word)
    pred = norm_pred(noun)
    gloss.setdefault(pred, f"{pred}(x) = x is a {noun}")
    pq.phrase.setdefault(pred, f"a {noun}")
    return pred


def pq_parse_sentence(s, pq: PQParser, gloss):
    """Returns ('fact', entity, [Lit,...]) (1 or 2 lits) or ('rule', RuleP)."""
    s = s.strip().rstrip(".").strip()
    m = re.fullmatch(r"(?:Every|Each) ([a-z][a-z-]*) is (.+)", s)
    if m:
        subj_p = _pq_noun_pred(pq, m.group(1), gloss)
        heads = _pq_head_lits(pq, m.group(2), gloss, s)
        return ("rule", RuleP([Lit(subj_p, ("?x",), False)], heads, True))
    # 'Each melodic vumpus is a shumpus' (adjective+noun body)
    m = re.fullmatch(r"(?:Every|Each) ([a-z][a-z-]*) ([a-z][a-z-]*) is (.+)", s)
    if m:
        adj_p, _ = pq.pred_of(m.group(1), gloss)
        noun_p = _pq_noun_pred(pq, m.group(2), gloss)
        heads = _pq_head_lits(pq, m.group(3), gloss, s)
        return (
            "rule",
            RuleP([Lit(adj_p, ("?x",), False), Lit(noun_p, ("?x",), False)], heads, True),
        )
    m = re.fullmatch(r"([A-Z][a-z-]*(?:es|s)) are (.+)", s)
    if m:
        subj_p = _pq_noun_pred(pq, m.group(1), gloss)
        heads = _pq_head_lits(pq, m.group(2), gloss, s)
        return ("rule", RuleP([Lit(subj_p, ("?x",), False)], heads, True))
    # 'Spicy impuses are wumpuses' (adjective + plural-noun body)
    m = re.fullmatch(r"([A-Z][a-z-]*) ([a-z][a-z-]*(?:es|s)) are (.+)", s)
    if m:
        adj_p, _ = pq.pred_of(m.group(1).lower(), gloss)
        noun_p = _pq_noun_pred(pq, m.group(2), gloss)
        heads = _pq_head_lits(pq, m.group(3), gloss, s)
        return (
            "rule",
            RuleP([Lit(adj_p, ("?x",), False), Lit(noun_p, ("?x",), False)], heads, True),
        )
    m = re.fullmatch(
        rf"Everything that is ((?:not )?(?:{ART} )?[a-z][a-z-]*) and ((?:not )?(?:{ART} )?[a-z][a-z-]*) is ((?:not )?(?:{ART} )?[a-z][a-z-]*)",
        s,
    )
    if m:
        b1 = pq.pred_of(m.group(1), gloss)
        b2 = pq.pred_of(m.group(2), gloss)
        h = pq.pred_of(m.group(3), gloss)
        return (
            "rule",
            RuleP(
                [Lit(b1[0], ("?x",), b1[1]), Lit(b2[0], ("?x",), b2[1])],
                Lit(h[0], ("?x",), h[1]),
                True,
            ),
        )
    m = re.fullmatch(r"([A-Z][a-zA-Z-]*) is (.+)", s)
    if m:
        entity = norm_const(m.group(1))
        DISPLAY_ENTITY.setdefault(entity, m.group(1))
        parts = re.split(r" and ", m.group(2))
        lits = []
        for part in parts:
            lits.extend(pq.lits_of(part.strip(), entity, gloss))
        if len(lits) > 2:
            raise Drop("pq_wide_conj", s)
        return ("fact", entity, lits)
    raise Drop("pq_unparsed", s)


def norm_text(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def convert_prontoqa_example(ex_id, ex):
    global DISPLAY_ENTITY
    DISPLAY_ENTITY = {}
    question = ex["question"].strip()
    query = ex["query"].strip()
    cot = list(ex["chain_of_thought"])
    answer_raw = ex.get("answer")
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", question) if s.strip()]
    pq = PQParser(sentences + cot + [query])
    gloss = {}
    premises = []
    prem_key = {}
    rule_by_text = {}
    for s in sentences:
        p = pq_parse_sentence(s, pq, gloss)
        if p[0] == "rule":
            premises.append(Premise(p[1].formula(), s))
            rule_by_text[norm_text(s)] = (len(premises), p[1])
        else:
            _, _entity, lits = p
            formula = lits[0].formula() if len(lits) == 1 else conj_formula(lits)
            premises.append(Premise(formula, s))
            prem_key[formula] = len(premises)

    q_m = re.fullmatch(r"(?:True or false:|Prove:)\s*(.+?)\.?", query)
    if not q_m:
        raise Drop("pq_bad_query", query)
    q_sent = q_m.group(1).strip()
    q_parsed = pq_parse_sentence(q_sent, pq, gloss)
    if q_parsed[0] != "fact":
        raise Drop("pq_bad_query_fact", query)
    entity = q_parsed[1]
    q_lits = q_parsed[2]
    q_formula = q_lits[0].formula() if len(q_lits) == 1 else conj_formula(q_lits)

    builder = ProofBuilder(premises)
    pending_impl = None  # (impl_line, grounded_head_formula, [grounded_body_formulas])
    last_line = None
    last_formula = None

    for s in cot:
        st = s.strip()
        key = norm_text(st)
        if key in rule_by_text:
            idx, rule = rule_by_text[key]
            g_body, g_heads, g_impl = rule.grounded(entity)
            impl_line = builder.have(g_impl)
            if impl_line is None:
                impl_line = builder.add(g_impl, f"AE,{idx}", st)
            head_formula = " & ".join(h.formula() for h in g_heads)
            pending_impl = (impl_line, head_formula, [l.formula() for l in g_body], g_body)
            last_line, last_formula = impl_line, g_impl
            continue
        p = pq_parse_sentence(st, pq, gloss)
        if p[0] != "fact":
            raise Drop("pq_cot_unparsed", st)
        _, _ent2, lits = p
        formula = lits[0].formula() if len(lits) == 1 else conj_formula(lits)
        if builder.have(formula):
            last_line, last_formula = builder.have(formula), formula
            pending_impl = None
            continue
        if formula in prem_key:
            last_line = builder.add(formula, f"R,{prem_key[formula]}", st)
            last_formula = formula
            pending_impl = None
            continue
        if pending_impl and formula == pending_impl[1]:
            impl_line, _g_head_f, g_body_fs, _body_lits = pending_impl
            if len(g_body_fs) == 1:
                body_ref = builder.have(g_body_fs[0])
            else:
                bodyconj = " & ".join(g_body_fs)
                body_ref = builder.have(bodyconj)
                if body_ref is None:
                    refs = [builder.have(f) for f in g_body_fs]
                    if any(r is None for r in refs):
                        raise Drop("pq_missing_body", st)
                    ent_disp = disp_entity(entity)
                    body_lits = pending_impl[3]
                    acc = g_body_fs[0]
                    acc_ref = refs[0]
                    for k in range(1, len(g_body_fs)):
                        acc = f"{acc} & {g_body_fs[k]}"
                        phrases = [
                            ("not " if l.neg else "") + pq.phrase.get(l.pred, l.pred.lower())
                            for l in body_lits[: k + 1]
                        ]
                        conj_nl = f"{ent_disp} is {' and '.join(phrases)}."
                        acc_ref = builder.add(acc, f"∧I,{acc_ref},{refs[k]}", conj_nl)
                    body_ref = acc_ref
            if body_ref is None:
                raise Drop("pq_missing_body", st)
            last_line = builder.add(formula, f"->E,{impl_line},{body_ref}", st)
            last_formula = formula
            pending_impl = None
            continue
        if len(lits) == 2:
            r1, r2 = builder.have(lits[0].formula()), builder.have(lits[1].formula())
            if r1 and r2:
                last_line = builder.add(formula, f"∧I,{r1},{r2}", st)
                last_formula = formula
                pending_impl = None
                continue
        if len(lits) == 1 and last_formula and " & " in last_formula and "->" not in last_formula:
            parts = last_formula.split(" & ")
            if formula in parts:
                last_line = builder.add(formula, f"∧E,{last_line}", st)
                last_formula = formula
                pending_impl = None
                continue
        raise Drop("pq_cot_step_unmapped", st)

    if last_formula is None:
        raise Drop("empty_proof")
    if answer_raw is None or str(answer_raw) == "True":
        ans = "True"
        if last_formula != q_formula:
            raise Drop("conclusion_mismatch", f"{last_formula} vs {q_formula}")
    else:
        ans = "False"
        if len(q_lits) != 1 or last_formula != q_lits[0].complement().formula():
            raise Drop("conclusion_mismatch_inv", f"{last_formula} vs {q_formula}")

    doc = Doc(
        source="prontoqa",
        id=ex_id,
        problem=f"{question}\n\n{query}",
        constants=sorted(f"{c} = {d}" for c, d in DISPLAY_ENTITY.items()),
        predicates=sorted(gloss.values()),
        premises=premises,
        answer=ans,
        depth=sum(1 for (_, _, j, _) in builder.lines if j.startswith("->E")),
        meta={},
    )
    finalize(doc, builder, last_formula)
    doc.conclusion_nl = clause(cot[-1]) + "."
    validate(doc)
    return doc


def build_prontoqa(raw_root: Path, out, limit=None):
    reasons = collections.Counter()
    kept = 0
    total = 0
    files = sorted(glob.glob(str(raw_root / "prontoqa_generated" / "*.json")))
    for path in files:
        with open(path) as fh:
            data = json.load(fh)
        fname = Path(path).stem
        for key, val in data.items():
            if not key.startswith("example"):
                continue
            ex = val.get("test_example", val) if isinstance(val, dict) else None
            if not isinstance(ex, dict) or "chain_of_thought" not in ex:
                continue
            total += 1
            if limit and kept >= limit:
                return kept, total, reasons
            try:
                doc = convert_prontoqa_example(f"{fname}::{key}", ex)
                doc.meta["file"] = fname
                emit(out, doc)
                kept += 1
            except Drop as d:
                reasons[d.reason] += 1
            except Exception as e:  # noqa: BLE001
                reasons[f"error_{type(e).__name__}"] += 1
    return kept, total, reasons


# --------------------------------------------------------------------------
# PARARULE-Plus converter (proofs re-derived by explicit forward chaining)
# --------------------------------------------------------------------------


class PRTheory:
    def __init__(self, context):
        self.sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", context.strip()) if s.strip()]
        self.premises = []
        self.facts = {}  # Lit -> premise idx
        self.rules = []  # (premise_idx, RuleP, text)
        self.entity_display = {}
        self.pred_gloss = {}
        for s in self.sentences:
            kind, payload = self.parse_sentence(s)
            self.premises.append(Premise(payload.formula(), s))
            if kind == "fact":
                self.facts.setdefault(payload, len(self.premises))
            else:
                self.rules.append((len(self.premises), payload, s))

    def _entity(self, raw):
        raw = raw.strip()
        core = re.sub(r"^[Tt]he ", "", raw)
        const = norm_const(core)
        disp = core if core[:1].isupper() else f"the {core}"
        self.entity_display.setdefault(const, disp)
        return const

    def _attr(self, word, neg):
        pred = norm_pred(word)
        self.pred_gloss.setdefault(pred, f"{pred}(x) = x is {word}")
        return pred, neg

    def parse_lit_phrase(self, subj, phrase):
        """phrase like 'is not big' | 'is big' | 'sees the rabbit'"""
        phrase = phrase.strip()
        m = re.fullmatch(r"is (not )?([a-z][a-z-]*)", phrase)
        if m:
            pred, neg = self._attr(m.group(2), bool(m.group(1)))
            return Lit(pred, (subj,), neg)
        m = re.fullmatch(r"([a-z][a-z-]*?)s? the ([a-z][a-z- ]*)", phrase)
        if m and m.group(1) != "i":
            verb = m.group(1) if m.group(1).endswith("s") else m.group(1) + "s"
            pred = norm_pred(verb)
            obj = self._entity("the " + m.group(2))
            self.pred_gloss.setdefault(pred, f"{pred}(x,y) = x {verb} y")
            return Lit(pred, (subj, obj), False)
        raise Drop("pr_bad_phrase", phrase)

    def parse_sentence(self, s):
        t = s.strip().rstrip(".").strip()
        m = re.fullmatch(r"If (?:something|someone) (.+?) then (?:it|they) (.+)", t)
        if m:
            body_txt, head_txt = m.group(1), m.group(2)
            body = []
            bm = re.fullmatch(r"is (not )?([a-z-]+)(?: and (not )?([a-z-]+))?", body_txt)
            if bm:
                p1 = self._attr(bm.group(2), bool(bm.group(1)))
                body.append(Lit(p1[0], ("?x",), p1[1]))
                if bm.group(4):
                    p2 = self._attr(bm.group(4), bool(bm.group(3)))
                    body.append(Lit(p2[0], ("?x",), p2[1]))
            else:
                body.append(self.parse_lit_phrase("?x", body_txt))
            hm = re.fullmatch(r"(?:is|are) (not )?([a-z-]+)", head_txt)
            if hm:
                hp = self._attr(hm.group(2), bool(hm.group(1)))
                head = Lit(hp[0], ("?x",), hp[1])
            else:
                head = self.parse_lit_phrase("?x", head_txt)
            return ("rule", RuleP(body, head, True))
        m = re.fullmatch(r"All ([a-z-]+) (?:animals|people) are ([a-z-]+)", t)
        if m:
            b = self._attr(m.group(1), False)
            h = self._attr(m.group(2), False)
            return ("rule", RuleP([Lit(b[0], ("?x",), False)], Lit(h[0], ("?x",), False), True))
        m = re.fullmatch(r"([A-Z][a-z-]+) (?:animals|people) are ([a-z-]+)", t)
        if m:
            b = self._attr(m.group(1).lower(), False)
            h = self._attr(m.group(2), False)
            return ("rule", RuleP([Lit(b[0], ("?x",), False)], Lit(h[0], ("?x",), False), True))
        m = re.fullmatch(
            r"((?:The [a-z-]+(?: (?!is\b|does\b|[a-z-]+s\b)[a-z-]+)?)|[A-Z][a-z-]+) (.+)", t
        )
        if m:
            subj = self._entity(m.group(1))
            lit = self.parse_lit_phrase(subj, m.group(2))
            return ("fact", lit)
        raise Drop("pr_unparsed", s)

    def display(self, lit: Lit) -> str:
        subj = disp_entity(lit.args[0])
        subj_s = subj[:1].upper() + subj[1:]
        if len(lit.args) == 1:
            adj = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", lit.pred).lower()
            return f"{subj_s} is {'not ' if lit.neg else ''}{adj}."
        verb = lit.pred.lower()
        obj = disp_entity(lit.args[1])
        return f"{subj_s} {verb} {obj}."


def pr_forward_chain(theory: PRTheory, max_iters=100):
    """Explicit-literal forward chaining (no NAF). Returns lit -> (depth, prov);
    prov is None for premises or (rule_premise_idx, rule, entity, [body_lits])."""
    derived = {lit: (0, None) for lit in theory.facts}
    entities = sorted({a for lit in theory.facts for a in lit.args})
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        for idx, rule, _text in theory.rules:
            for ent in entities:
                body = [l.subst(ent) for l in rule.body]
                if all(b in derived for b in body):
                    head = rule.head.subst(ent)
                    if head in derived:
                        continue
                    d = 1 + max((derived[b][0] for b in body), default=0)
                    derived[head] = (d, (idx, rule, ent, body))
                    changed = True
    return derived


def convert_pararule_question(theory: PRTheory, derived, ctx_id, q):
    global DISPLAY_ENTITY
    DISPLAY_ENTITY = theory.entity_display
    text = q["text"].strip()
    label = str(q["label"]).strip().lower()
    kind, target = theory.parse_sentence(text)
    if kind != "fact":
        raise Drop("pr_question_not_fact", text)
    if label == "true":
        goal, ans = target, "True"
    elif label == "false":
        goal, ans = target.complement(), "False"
    else:
        raise Drop("pr_unknown_label", label)
    if goal not in derived:
        raise Drop("not_derivable_explicit", goal.formula())

    builder = ProofBuilder(theory.premises)

    def emit_lit(lit):
        f = lit.formula()
        existing = builder.have(f)
        if existing:
            return existing
        depth, prov = derived[lit]
        if prov is None:
            idx = theory.facts[lit]
            return builder.add(f, f"R,{idx}", f"We are given: {theory.display(lit)}")
        idx, rule, ent, body = prov
        body_lines = [(emit_lit(b), b) for b in body]
        line, _ = emit_rule_application(builder, idx, rule, ent, body_lines, theory.display)
        return line

    emit_lit(goal)
    depth = derived[goal][0]

    doc = Doc(
        source="pararule",
        id=str(ctx_id) + "::" + str(q.get("id", "q")),
        problem=" ".join(theory.sentences) + f"\n\nTrue or false: {text}",
        constants=sorted(f"{c} = {d}" for c, d in theory.entity_display.items()),
        predicates=sorted(theory.pred_gloss.values()),
        premises=theory.premises,
        answer=ans,
        depth=depth,
        meta={
            "qcat": (q.get("meta") or {}).get("QCat", ""),
            "qdep_gold": (q.get("meta") or {}).get("QDep", ""),
        },
    )
    finalize(doc, builder, goal.formula())
    doc.conclusion_nl = theory.display(goal)
    validate(doc)
    return doc


def build_pararule(raw_root: Path, out, limit=None):
    reasons = collections.Counter()
    kept = 0
    total = 0
    files = sorted(glob.glob(str(raw_root / "pararule" / "*_train.jsonl"))) + sorted(
        glob.glob(str(raw_root / "pararule" / "*_dev.jsonl"))
    )
    for path in files:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                try:
                    theory = PRTheory(rec["context"])
                    derived = pr_forward_chain(theory)
                except (Drop, Exception) as d:  # noqa: BLE001
                    r = d.reason if isinstance(d, Drop) else f"error_{type(d).__name__}"
                    for _q in rec.get("questions", []):
                        total += 1
                        reasons[f"theory_{r}"] += 1
                    continue
                for q in rec.get("questions", []):
                    total += 1
                    if limit and kept >= limit:
                        return kept, total, reasons
                    try:
                        doc = convert_pararule_question(theory, derived, rec["id"], q)
                        doc.meta["file"] = Path(path).name
                        emit(out, doc)
                        kept += 1
                    except Drop as d:
                        reasons[d.reason] += 1
                    except Exception as e:  # noqa: BLE001
                        reasons[f"error_{type(e).__name__}"] += 1
    return kept, total, reasons


# --------------------------------------------------------------------------
# Emission / dedup / audit
# --------------------------------------------------------------------------


def emit(out_fh, doc: Doc):
    formal, nl = render_pair(doc)
    row = {
        "source": doc.source,
        "id": doc.id,
        "depth": doc.depth,
        "answer": doc.answer,
        "n_premises": len(doc.premises),
        "n_proof_lines": len(doc.proof_formal),
        "problem": doc.problem,
        "formal_doc": formal,
        "nl_doc": nl,
        "meta": doc.meta,
    }
    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_for_dedup(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def load_test_texts(raw_root: Path):
    """Held-out texts we must not overlap with: ProofWriter OWA test splits,
    PARARULE-Plus test splits, all of FOLIO (eval-only). Returns
    (exact_problem_set, unique_theory_list)."""
    exact = set()
    theories = set()
    base = raw_root / "proofwriter" / "proofwriter-dataset-V2020.12.3" / "OWA"
    for depth_dir in ["depth-0", "depth-1", "depth-2", "depth-3", "depth-5", "birds-electricity"]:
        p = base / depth_dir / "meta-test.jsonl"
        if not p.exists():
            continue
        with open(p) as fh:
            for line in fh:
                rec = json.loads(line)
                th = norm_for_dedup(rec["theory"])
                theories.add(th)
                for q in rec.get("questions", {}).values():
                    exact.add(th + " true or false " + norm_for_dedup(q["question"]))
    for p in sorted(glob.glob(str(raw_root / "pararule" / "*_test.jsonl"))):
        with open(p) as fh:
            for line in fh:
                rec = json.loads(line)
                th = norm_for_dedup(rec["context"])
                theories.add(th)
                for q in rec.get("questions", []):
                    exact.add(th + " true or false " + norm_for_dedup(q["text"]))
    for p in sorted(glob.glob(str(raw_root / "folio" / "*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prem = rec.get("premises")
                if isinstance(prem, list):
                    prem = " ".join(prem)
                th = norm_for_dedup(prem or "")
                theories.add(th)
                exact.add(th + " true or false " + norm_for_dedup(rec.get("conclusion") or ""))
    return exact, sorted(theories)


def ngrams(tokens, n):
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def cmd_dedup(args):
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    exact, theories = load_test_texts(raw_root)
    n = args.ngram
    gram_index = {}
    theory_gram_counts = []
    for di, t in enumerate(theories):
        gs = ngrams(t.split(), n)
        theory_gram_counts.append(max(len(gs), 1))
        for g in gs:
            gram_index.setdefault(g, []).append(di)
    stats = {}
    for path in sorted(glob.glob(str(out_root / "corpus" / "*.jsonl"))):
        if path.endswith(".dedup.jsonl"):
            continue
        src = Path(path).stem
        kept, exact_hits, near_hits = 0, 0, 0
        out_path = out_root / "corpus" / f"{src}.dedup.jsonl"
        with open(path) as fh, open(out_path, "w") as out:
            for line in fh:
                row = json.loads(line)
                t = norm_for_dedup(row["problem"])
                if t in exact:
                    exact_hits += 1
                    continue
                gs = ngrams(t.split(), n)
                hit_counts = collections.Counter()
                for g in gs:
                    for di in gram_index.get(g, ()):
                        hit_counts[di] += 1
                dropped = False
                if gs and hit_counts:
                    di, hits = hit_counts.most_common(1)[0]
                    denom = min(len(gs), theory_gram_counts[di])
                    if hits / denom >= args.threshold:
                        near_hits += 1
                        dropped = True
                if not dropped:
                    out.write(line)
                    kept += 1
        stats[src] = {"kept": kept, "exact_dropped": exact_hits, "near_dup_dropped": near_hits}
        print(src, stats[src])
    with open(out_root / "dedup_stats.json", "w") as fh:
        json.dump(
            {
                "ngram": n,
                "threshold": args.threshold,
                "test_problems_exact": len(exact),
                "test_theories_ngram": len(theories),
                "stats": stats,
            },
            fh,
            indent=2,
        )


def get_token_counter(tokenizer_path):
    if tokenizer_path and Path(tokenizer_path).exists():
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tokenizer_path))
            return ("qwen2.5", lambda s: len(tok.encode(s).ids))
        except Exception:
            pass
    return ("whitespace-proxy", lambda s: len(s.split()))


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def cmd_audit(args):
    out_root = Path(args.out_root)
    tok_name, count_tokens = get_token_counter(args.tokenizer)
    rng = random.Random(3407)
    lines_md = [
        "# Real-data paired logic/NL corpus - audit",
        "",
        f"Tokenizer for token counts: **{tok_name}**",
        "",
    ]
    build_stats = {}
    bs_path = out_root / "build_stats.json"
    if bs_path.exists():
        build_stats = json.loads(bs_path.read_text())
    dd_path = out_root / "dedup_stats.json"
    if dd_path.exists():
        lines_md += [
            "## Dedup vs held-out splits (ProofWriter OWA test incl. birds-electricity, PARARULE-Plus test, FOLIO v0 all)",
            "",
            "```json",
            dd_path.read_text().strip(),
            "```",
            "",
        ]
    grand = {"formal": 0, "nl": 0, "docs": 0}
    for path in sorted(glob.glob(str(out_root / "corpus" / "*.dedup.jsonl"))):
        src = Path(path).stem.replace(".dedup", "")
        ftoks, ntoks = [], []
        depths = collections.Counter()
        answers = collections.Counter()
        reservoir = []
        n_rows = 0
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                n_rows += 1
                ftoks.append(count_tokens(row["formal_doc"]))
                ntoks.append(count_tokens(row["nl_doc"]))
                depths[row["depth"]] += 1
                answers[row["answer"]] += 1
                if len(reservoir) < 5:
                    reservoir.append(row)
                else:
                    j = rng.randrange(n_rows)
                    if j < 5:
                        reservoir[j] = row
        ftoks_s, ntoks_s = sorted(ftoks), sorted(ntoks)
        tot_f, tot_n = sum(ftoks), sum(ntoks)
        grand["formal"] += tot_f
        grand["nl"] += tot_n
        grand["docs"] += n_rows
        lines_md += [f"## Source: {src}", ""]
        if src in build_stats:
            bs = build_stats[src]
            lines_md += [
                f"- examples scanned: {bs['total']}, converted+validated: {bs['kept']} ({100*bs['kept']/max(bs['total'],1):.1f}%)",
                f"- drop reasons: `{json.dumps(bs['reasons'], sort_keys=True)}`",
            ]
        lines_md += [
            f"- kept after dedup: **{n_rows}** paired docs",
            f"- answers: {dict(answers)}",
            f"- depth distribution: {dict(sorted(depths.items()))}",
            f"- formal doc tokens: total {tot_f:,}; mean {tot_f/max(n_rows,1):.0f}; p50 {pct(ftoks_s,50)}; p90 {pct(ftoks_s,90)}; max {ftoks_s[-1] if ftoks_s else 0}",
            f"- NL doc tokens: total {tot_n:,}; mean {tot_n/max(n_rows,1):.0f}; p50 {pct(ntoks_s,50)}; p90 {pct(ntoks_s,90)}; max {ntoks_s[-1] if ntoks_s else 0}",
            "",
        ]
        for i, row in enumerate(reservoir, 1):
            lines_md += [
                f"### {src} random example {i} (id={row['id']}, depth={row['depth']})",
                "",
                "FORMAL DOC:",
                "```",
                row["formal_doc"],
                "```",
                "NL DOC:",
                "```",
                row["nl_doc"],
                "```",
                "",
            ]
        sample_path = out_root / "samples" / f"{src}_sample100.jsonl"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with open(path) as fh:
            rows = fh.readlines()
        rng2 = random.Random(1234)
        picks = rows if len(rows) <= 100 else rng2.sample(rows, 100)
        with open(sample_path, "w") as fh:
            fh.writelines(picks)
    lines_md += [
        "## Totals",
        "",
        f"- paired docs: {grand['docs']:,}",
        f"- formal tokens: {grand['formal']:,}",
        f"- NL tokens: {grand['nl']:,}",
        f"- combined (formal+NL): {grand['formal']+grand['nl']:,}",
        "",
    ]
    (out_root / "audit.md").write_text("\n".join(lines_md))
    print("wrote", out_root / "audit.md")


def cmd_build(args):
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    (out_root / "corpus").mkdir(parents=True, exist_ok=True)
    out_path = out_root / "corpus" / f"{args.source}.jsonl"
    builders = {"proofwriter": build_proofwriter, "prontoqa": build_prontoqa, "pararule": build_pararule}
    with open(out_path, "w") as out:
        kept, total, reasons = builders[args.source](raw_root, out, limit=args.limit)
    print(f"{args.source}: kept {kept}/{total}")
    for r, c in reasons.most_common():
        print(f"  {r}: {c}")
    stats_path = out_root / "build_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats[args.source] = {"kept": kept, "total": total, "reasons": dict(reasons)}
    stats_path.write_text(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=["proofwriter", "prontoqa", "pararule"], required=True)
    b.add_argument("--raw-root", required=True)
    b.add_argument("--out-root", required=True)
    b.add_argument("--limit", type=int, default=None)
    b.set_defaults(func=cmd_build)
    d = sub.add_parser("dedup")
    d.add_argument("--raw-root", required=True)
    d.add_argument("--out-root", required=True)
    d.add_argument("--ngram", type=int, default=10)
    d.add_argument("--threshold", type=float, default=0.8)
    d.set_defaults(func=cmd_dedup)
    a = sub.add_parser("audit")
    a.add_argument("--out-root", required=True)
    a.add_argument("--tokenizer", default=str(SCRIPT_DIR / "qwen25_tokenizer.json"))
    a.set_defaults(func=cmd_audit)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
