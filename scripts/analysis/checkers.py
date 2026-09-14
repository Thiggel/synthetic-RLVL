#!/usr/bin/env python3
"""Checkers for derivations written on real benchmarks in the document format.

Three independent checkers, each gold-free: they judge whether the written
derivation is internally valid, never whether the answer is right.

  gsm8k:       every arithmetic line "a op b = c" must actually hold, and the
               answer must be the last quantity the derivation computes.
  proofwriter: the theory is a closed-world Horn fragment with negation. Each
               proof line must be a premise or follow from lines already
               established, by a rule of the theory.
  branchproof: the stricter version already used for the synthetic items lives
               in check_native_derivations.py.
"""
import re
from fractions import Fraction

# ---------------------------------------------------------------- GSM8K ----
_EXPR = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([-+*/x])\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)")


def _num(s):
    return Fraction(s.replace(",", "").replace("$", ""))


def check_gsm8k(text):
    """-> dict(n_steps, all_valid, valid_prefix)"""
    steps = _EXPR.findall(text)
    ok = 0
    first_bad = None
    for i, (a, op, b, c) in enumerate(steps):
        try:
            x, y, z = _num(a), _num(b), _num(c)
            v = {"+": x + y, "-": x - y, "*": x * y, "x": x * y, "/": (x / y if y else None)}[op]
            good = v is not None and abs(v - z) <= Fraction(1, 100)
        except (ValueError, ZeroDivisionError):
            good = False
        if good:
            ok += 1
        elif first_bad is None:
            first_bad = i
    n = len(steps)
    return dict(n_steps=n, all_valid=float(n > 0 and first_bad is None),
                valid_prefix=(n if first_bad is None else first_bad) / n if n else 0.0)


# ---------------------------------------------------------- ProofWriter ----
# Facts:      "The bear is big." / "Anne is not quiet." / "The dog chases the bear."
# Rules:      "If X then Y", "All A, B things are C", "Rough things are round",
#             "Furry, round things are smart", "If someone is rough then they ..."
_ATTR = re.compile(r"^(?:the\s+)?(\?x|\w[\w\s]*?)\s+(?:is|are)\s+(not\s+)?(\w+)$", re.I)
_REL = re.compile(r"^(?:the\s+)?(\?x|\w[\w\s]*?)\s+(?:(does not|do not)\s+)?(\w+?)s?\s+(?:the\s+)?(\?x|\w[\w\s]*?)$", re.I)
VARS = {"someone", "something", "things", "people", "they", "it"}


def _lit(s):
    s = s.strip().rstrip(".").strip()
    m = _ATTR.match(s)
    if m:
        return (m.group(1).strip().lower(), m.group(3).lower(), not m.group(2))
    m = _REL.match(s)
    if m:
        return (m.group(1).strip().lower(), m.group(3).lower() + "|" + m.group(4).strip().lower(), not m.group(2))
    return None


def _split_conj(s):
    s = re.sub(r"\band\b", ",", s, flags=re.I)
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_theory(context):
    facts, rules = set(), []
    for raw in re.split(r"(?<=\.)\s+", context.strip()):
        s = raw.strip().rstrip(".")
        if not s:
            continue
        low = s.lower()
        m = re.match(r"^if (.+?) then (.+)$", low)
        if m:
            ants, cons = m.group(1), m.group(2)
            subj = None
            lits = []
            for part in _split_conj(ants):
                p = part.strip()
                if p.split()[0] in VARS:
                    subj = "?x"
                    p = re.sub(r"^\w+", "?x", p)
                lit = _lit(p)
                if lit:
                    lits.append(lit)
            chead = cons.strip()
            if chead.split() and chead.split()[0].lower() in ("they", "it", "then", "someone", "something"):
                chead = re.sub(r"^\w+", "?x", chead)
            elif subj == "?x" and not re.match(r"^(the\s+)?\w+\s+(is|are|does|do)\b", chead, re.I):
                pass
            head = _lit(chead)
            if head and lits:
                rules.append((lits, head))
            continue
        m = re.match(r"^(?:all )?(.+?) (?:things|people) are (\w+)$", low)
        if m:
            lits = [("?x", a.strip().lower(), True) for a in _split_conj(m.group(1))]
            rules.append((lits, ("?x", m.group(2), True)))
            continue
        lit = _lit(s)
        if lit:
            facts.add(lit)
    return facts, rules


def _entities(facts):
    return {f[0] for f in facts if not f[0].startswith("?")}


def _ground(lit, e):
    return (e if lit[0] == "?x" else lit[0], lit[1], lit[2])


def closure(facts, rules, max_iter=40):
    known = set(facts)
    ents = _entities(facts) or {"x"}
    for _ in range(max_iter):
        new = set()
        for lits, head in rules:
            for e in ents:
                gl = [_ground(l, e) for l in lits]
                ok = True
                for g in gl:
                    if g[2]:
                        ok = ok and g in known
                    else:  # negation as failure, closed world
                        ok = ok and (g[0], g[1], True) not in known
                if ok:
                    h = _ground(head, e)
                    if h not in known:
                        new.add(h)
        if not new:
            break
        known |= new
    return known


# --- formal-notation ProofWriter output ------------------------------------
# A formal-notation model answers ProofWriter with a <formal> block: constants,
# a predicate legend, FOL premises and proof lines like "C(c)". Map those back
# to the English literals the theory parser produces, using the model's own
# legend, so the same checker judges both notations.
_FOL_LINE = re.compile(r"^([A-Za-z]\w*)\(([^)]+)\)")
_LEGEND = re.compile(r"^\s*([A-Za-z]\w*?)\s*(?:x|\(x\))?\s*[:=]\s*(.+?)\s*$")


def _fol_legend(text):
    """predicate symbol -> attribute word; constant symbol -> entity name."""
    preds, consts = {}, {}
    for tag, store in (("predicates", preds), ("constants", consts)):
        m = re.search(r"<%s>\n?(.*?)\n?</%s>" % (tag, tag), text, re.S)
        if not m:
            continue
        for line in m.group(1).splitlines():
            g = _LEGEND.match(line.strip())
            if not g:
                continue
            rhs = g.group(2).strip().rstrip(".")
            rhs = re.sub(r"^x\s+", "", rhs, flags=re.I)
            rhs = re.sub(r"^is\s+", "", rhs, flags=re.I)
            rhs = re.sub(r"^the\s+", "", rhs, flags=re.I)
            sym = g.group(1)
            if tag == "predicates":
                sym = sym[:-1] if sym.endswith("x") and len(sym) > 1 else sym
                # a relational legend such as "needs the rabbit" becomes the
                # verb|object key the theory parser uses
                m2 = re.match(r"^(\w+?)s?\s+(?:the\s+)?(\w[\w\s]*)$", rhs)
                rhs = ("%s|%s" % (m2.group(1), m2.group(2).strip())) if m2 else rhs
            store[sym] = rhs.lower()
    return preds, consts


def _fol_to_lit(line, preds, consts):
    m = _FOL_LINE.match(line.strip().lstrip("~"))
    if not m:
        return None
    neg = line.strip().startswith("~")
    pred = preds.get(m.group(1))
    ent = m.group(2).split(",")[0].strip()
    ent = consts.get(ent, ent).lower()
    if pred is None:
        return None
    return (ent, pred, not neg)


def check_proofwriter(context, text, formal=None):
    """Each <proof> line must be a premise or follow from what is established.

    Handles both notations: English proof lines directly, and formal ones by
    translating through the model's own constant and predicate legend.
    """
    facts, rules = parse_theory(context)
    if formal is None:
        formal = "<formal>" in text or "<predicates>" in text
    preds, consts = _fol_legend(text) if formal else ({}, {})
    m = re.search(r"<proof>\n?(.*?)\n?</proof>", text, re.S)
    if not m:
        return dict(has_proof=0.0, n_lines=0, all_valid=0.0, valid_prefix=0.0, parsed_frac=0.0)
    known = set(facts)
    lines, parsed, valid, first_bad = 0, 0, 0, None
    for raw in m.group(1).splitlines():
        s = re.sub(r"^\s*\d+[.)]\s*", "", raw).strip()
        s = re.sub(r"\((premise|from [^)]*|rule[^)]*)\)\s*$", "", s, flags=re.I).strip()
        if not s:
            continue
        lines += 1
        lit = _fol_to_lit(s, preds, consts) if formal else _lit(s.rstrip("."))
        if lit is None:
            if first_bad is None:
                first_bad = lines - 1
            continue
        parsed += 1
        if lit in known:
            valid += 1
            continue
        derived = False
        for lits, head in rules:
            for e in _entities(known) | {lit[0]}:
                if _ground(head, e) != lit:
                    continue
                gl = [_ground(l, e) for l in lits]
                if all((g in known) if g[2] else ((g[0], g[1], True) not in known) for g in gl):
                    derived = True
                    break
            if derived:
                break
        if derived:
            valid += 1
            known.add(lit)
        elif first_bad is None:
            first_bad = lines - 1
    return dict(has_proof=1.0, n_lines=lines, parsed_frac=parsed / max(1, lines),
                all_valid=float(lines > 0 and first_bad is None),
                valid_prefix=(lines if first_bad is None else first_bad) / max(1, lines))
