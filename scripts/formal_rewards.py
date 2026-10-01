"""Checker-based rewards for the Stage-2 GRPO arms (docs/research_plan.md).

A completion is scored once with eval_formal_bench_vllm.score (rlvl.check, loose
and strict) and cached; each reward function reads a component of that record.

Components (all 0/1):
  correct      the final answer matches the reference (Answer: line, else the
               proof's `ans`, else <answer>/\\boxed{}), as in the tagged eval
  grammatical  the proof parses (no fatal parse error); 0 without a proof
  valid        Stage-2 validity: rlvl.check strict ok AND grounded (>= 1
               `given`, no quote error) AND the proof's own `ans` agrees with the
               Answer: line (or there is no Answer: line). Grounding and answer
               agreement close the hacks seen in the 2026-09 RL runs (valid but
               irrelevant proofs; proofs that conclude a marker, not the answer).
               It also needs >= 1 checked derived (non-given) line the conclusion
               depends on, and the conclusion must not restate a `given` formula: G5 at
               step 50 won 29 of its 33 in-system gate proofs on yes/no items by quoting the
               claim as a `given` and answering `ans yes` (2026-09-28 17:30). Tautological
               derived lines (`a = a`) do not count as that derived line (2026-09-28 19:40).
  in_system    valid AND the proof's `ans` equals the reference
Arms (the primary reward; the other components are logged with weight 0):
  G0/G1 correct            G2 correct_x_valid        G3 gvc = (grammatical + valid + correct) / 3
  G4 valid
  G5 lines   dense line credit, the user's "(#grammatical lines + #valid lines) * (1 + correct)"
             hardened against padding (see line_stats):
             ((g + v) / 2 + valid) * (1 + correct) / 4  in [0, 1]
             g, v = min(1, n / LINE_CAP) for the parsed / checking derived (non-given)
             lines that the conclusion depends on. The valid bonus keeps a complete valid proof above
             any partial one.
  lines_fmt  G5c: lines * format_ok (one </proof>, then only the Answer: line)
  cvf_fmt    cvf * format_ok. cvf ignores what follows </proof>; under it G10 drifted from `Answer: 3` to
             `Answer yes ; 3` (format_ok 0 on every rollout from step ~540, still rewarded) and then into
             `</proof>\n<formal>\n<prompt copy>` loops (2026-10-01, reports/2026-09-28_stage2_gate.md)
  lines_raw  the literal formula on the same line counts, (n_parsed + n_ok) * (1 + correct),
             for comparison only (rewards length)
"""
from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_formal_bench_vllm import match, score  # noqa: E402
from eval_formal_vllm import extract, faithful_given, givens  # noqa: E402

_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = 65536
LINE_CAP = 8
_STEP = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*?)\s*;\s*(.*)$")
_ANS = re.compile(r"^\s*ans\b.*;\s*(\d+(?:\.\d+)*)\s*$")
_LABEL = re.compile(r"^\d+(?:\.\d+)*$")
_REL = re.compile(r"<->|->|<=|>=|!=|=")
_TAIL = re.compile(r"\s*Answer:[^\n]*\s*")


def tautology(f: str) -> bool:
    """`a = a`, `p <-> p`, `p -> p`, `x <= x`: the first top-level relation has equal sides."""
    depth = 0
    for i, ch in enumerate(f):
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth == 0:
            m = _REL.match(f, i)
            if m:
                lhs, rhs = f[:i], f[m.end():]
                return m.group() != "!=" and "".join(lhs.split()) == "".join(rhs.split()) != ""
    return False


_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*/\s*(\d+))?")
_WORDS = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
          "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
          "million": 10**6, "dozen": 12, "half": Fraction(1, 2), "quarter": Fraction(1, 4), "twice": 2, "double": 2,
          "triple": 3, "thrice": 3, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
          "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "once": 1, "pair": 2, "single": 1}
FREE_NUMBERS = {Fraction(0), Fraction(1)}


def _numbers(text: str, quote: bool) -> set:
    """Numeric values in a formula, or (quote=True) every value a quote can state: digits, decimals,
    fractions, percentages (25% -> 25 and 1/4), number words (half, dozen, third -> 1/3 and 3)."""
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    out = set()
    for m in _NUM.finditer(t):
        a, b = m.groups()
        try:
            v = Fraction(a) / Fraction(b) if b else Fraction(a)
        except (ValueError, ZeroDivisionError):
            continue
        out.add(v)
        if quote:
            out |= {Fraction(a)} | ({Fraction(b)} if b else set()) | {v / 100}
    if quote:
        for w in re.findall(r"[a-z]+", t.lower()):
            if w in _WORDS:
                v = Fraction(_WORDS[w])
                out |= {v, 1 / v} if v else {v}
    return out


# rules whose line rlvl accepts without deriving it: they earn no checked-line credit, count as
# premises for circularity, and are not tautology-exempt (G6 @90: `cuts = 6 ; know`, then `subst`)
TRUST = {"given", "know", "def", "obs", "assume"}
# unit constants background `know` lines may use without the prompt stating them (days = weeks * 7, ...)
UNIT_CONSTANTS = " 7 12 24 60 100 1000"


def premise_numbers_ok(formula: str, quote: str) -> bool:
    """Every number in a `given` formula is stated (by value) in its quote. A cheap partial
    premise check: it rejects `x = 42 ; given "<any sentence>"` (a guessed answer smuggled in as
    a premise), not wrong relations between stated quantities."""
    return _numbers(formula, False) <= _numbers(quote, True) | FREE_NUMBERS


def format_ok(completion: str) -> bool:
    """One `</proof>`, followed only by the Answer: line. Late G5 (steps 60-99) ran on after
    </proof> (repeated </proof>, "Now, we write the final answer..." loops) and 83% had no Answer: line."""
    parts = completion.split("</proof>")
    return len(parts) == 2 and _TAIL.fullmatch(parts[1]) is not None


def line_stats(prompt: str, completion: str) -> dict:
    """Line counts for the dense arm, restricted to the lines the conclusion depends on.

    Root = the line cited by `ans`, else the last step line. Ancestors follow the
    label citations after the rule name (quotes excluded). Counting only ancestors
    (distinct formulas) gives no credit for padding: repeated `given` lines, or free
    `calc`/`refl` lines that cite nothing the conclusion uses.
      n_steps    step lines in the proof text
      n_parsed   ancestor lines the parser reached (before a parse error)
      n_ok       derived ancestor lines (rule != given) that check (rlvl strict per-line ok);
                 `given` lines always check (grounding is a substring test), so counting
                 them would pay for citing premises from one bogus final line
      circular   1 if the conclusion's formula is the formula of an ancestor `given` line
      n_taut     ancestor derived lines that are tautologies (`child = child ; subst 2 3`); they
                 count toward neither n_parsed nor n_ok (late G5 chained them for line credit)
    The body lines of a block (labels L.1, L.2, ... under L) are ancestors of L, and L checks only
    if its whole body does.
    """
    import rlvl
    zero = {"n_steps": 0, "n_parsed": 0, "n_ok": 0, "circular": 0, "n_taut": 0, "frac_parsed": 0.0,
            "frac_ok": 0.0, "n_prem_bad": 0}
    s = completion.find("<proof>\n")
    if s < 0:
        return zero
    e = completion.find("</proof>", s)
    body = completion[s + len("<proof>\n"): e if e >= 0 else len(completion)]
    cites, formula, rule, quote, order, root = {}, {}, {}, {}, [], None
    for ln in body.split("\n"):
        m = _ANS.match(ln)
        if m:
            root = m.group(1)
            break
        m = _STEP.match(ln)
        if m:
            lab, f, just = m.groups()
            q = re.search(r'"([^"]*)"', just)
            quote[lab] = q.group(1) if q else ""
            just = just.split('"', 1)[0].split()
            cites[lab] = [t for t in just[1:] if _LABEL.match(t)]
            rule[lab] = just[0] if just else ""
            formula[lab] = f
            order.append(lab)
    if not order:
        return zero
    root = root if root in cites else order[-1]
    anc, stack = set(), [root]
    while stack:
        x = stack.pop()
        if x in anc or x not in cites:
            continue
        anc.add(x)
        stack += cites[x] + [y for y in order if y.startswith(x + ".")]  # a block's body is part of it
    try:
        rep = rlvl.check(prompt, body, strict=True)
    except BaseException:  # pyo3 PanicException is not an Exception
        return {**zero, "n_steps": len(order)}
    parsed, ok, taut = {}, {}, set()
    steps = {r["label"]: r for r in rep.get("lines") or [] if r.get("kind") == "step" and r.get("label") in anc}
    # a block line (`contra`, `cases`, ...) checks only if every line of its body was parsed and checks:
    # rlvl marks the header ok even when its body is wrong or cut by a parse error (G6 @80-95 hack)
    body_ok = lambda lab: all(y in steps and steps[y].get("ok") for y in anc if y.startswith(lab + "."))
    norm = lambda f: "".join(f.split())
    for lab, r in steps.items():
        f = formula.get(lab)
        # premise-free arithmetic (`4 + 1 = 5 ; calc`, citing nothing) is free padding, like `a = a` (G6 @93)
        ground = rule.get(lab) == "calc" and not cites.get(lab) and not re.search(r"[A-Za-z_]", f or "")
        # a line restating a formula it cites derives nothing: rlvl accepts `3 cos=2 ; subst 2 1` locally
        # although lines 1-2 (`cos=2 ; calc`) fail (G6 @150: one such shell on 72% of gate items)
        restate = any(norm(formula.get(c, "")) == norm(f or "") for c in cites.get(lab, []))
        if rule.get(lab) not in TRUST and (tautology(f) or ground or restate):
            taut.add(f)
            continue
        parsed[f] = True
        if rule.get(lab) not in TRUST:  # a derived formula checks only if every line deriving it checks
            ok[f] = ok.get(f, True) and bool(r.get("ok") and body_ok(lab))
    circ = any(rule[x] in TRUST and norm(formula[x]) == norm(formula[root]) for x in anc)
    # fractions over the distinct ancestor formulas: parsed / all, checked derived / derived (tautologies
    # count in the denominator only). No length incentive; the minimum is a 2-line proof.
    anc_f = {formula[x] for x in anc}
    # trusted lines earn nothing; one rlvl rejects counts as a failed derived line (G6 @92:
    # `girls = 3 / 5 * 500 ; def` smuggles a premise in). Quoted premises (given, obs) must state
    # their numbers; unquoted ones (know, def) may use only numbers of the prompt.
    der_f = {formula[x] for x in anc if rule[x] not in TRUST or not (x in steps and steps[x].get("ok"))}
    prem_bad = sum(not premise_numbers_ok(formula[x], quote[x] if rule[x] in ("given", "obs") else prompt + UNIT_CONSTANTS)
                   for x in anc if rule[x] in TRUST - {"assume"})
    return {"n_steps": len(order), "n_parsed": len(parsed), "n_ok": sum(ok.values()), "circular": int(circ),
            "n_taut": len(taut), "frac_parsed": len(parsed) / len(anc_f),
            "frac_ok": sum(ok.values()) / len(der_f) if der_f else 0.0, "n_prem_bad": prem_bad}


def components(rec: dict, completion: str) -> dict:
    key = (rec["id"], completion)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        row = score(rec, completion)
    except BaseException as e:  # a checker crash (incl. a Rust panic, a BaseException) must not kill the run
        row = {"correct": False, "grammatical": False, "valid": False, "grounded": False, "sys_answer": None,
               "pred_answer": None, "has_proof": False, "error": f"crash: {e!r}"}
    agree = True
    if row.get("pred_answer") is not None and row.get("sys_answer") is not None:
        try:  # an unparseable Answer: line ("nan" gold) disagrees
            agree = match(str(row["sys_answer"]), {**rec, "gold": _gold_like(rec, row["pred_answer"])}) == 1.0
        except Exception:
            agree = False
    ls = line_stats(rec["prompt"], completion)
    valid = bool(row["valid"] and row["grounded"] and row.get("sys_answer") is not None and agree
                 and ls["n_ok"] >= 1 and not ls["circular"])
    valid_prem = valid and ls["n_prem_bad"] == 0
    in_sys = bool(valid and rec.get("system_answerable")
                  and match(str(row["sys_answer"]), rec) == 1.0)
    out = {"correct": float(bool(row["correct"])), "grammatical": float(bool(row["grammatical"])),
           "valid": float(valid), "in_system": float(in_sys), "has_proof": float(bool(row["has_proof"])),
           "valid_strict": float(bool(row["valid"])), "format_ok": float(format_ok(completion)),
           **{k: float(v) for k, v in ls.items()}}
    g, v = min(1.0, ls["n_parsed"] / LINE_CAP), min(1.0, ls["n_ok"] / LINE_CAP)
    out["lines"] = ((g + v) / 2 + valid) * (1 + out["correct"]) / 4
    out["lines_fmt"] = out["lines"] * out["format_ok"]
    out["valid_prem"] = float(valid_prem)
    # the user's "%valid lines + %grammatical lines + correct", as is and hardened
    out["frac"] = (ls["frac_parsed"] + ls["frac_ok"] + out["correct"]) / 3
    # premise check: exact faithfulness (every `given` is a gold form of the quoted sentence) on generator
    # items, which carry their sentences; the numeric check elsewhere (Dolci has no gold formalization)
    if rec.get("sentences_json"):
        sents, proof = json.loads(rec["sentences_json"]), extract(completion)["proof"] or ""
        prem = float(all(faithful_given(f, q, sents) for _, f, q in givens(proof)))
    else:
        prem = float(ls["n_prem_bad"] == 0)
    out["prem_ok"] = prem
    out["frac_hard"] = ((ls["frac_parsed"] + ls["frac_ok"]) * prem * out["format_ok"] * (1 - ls["circular"])
                        + out["correct"]) / 3
    out["lines_raw"] = (ls["n_parsed"] + ls["n_ok"]) * (1 + out["correct"])
    _CACHE[key] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out


def _gold_like(rec: dict, pred: str) -> str:
    """The Answer: line as a reference of the item's answer type (for ans/Answer agreement)."""
    from eval_formal_bench_vllm import YN, _clean, as_number
    t = rec["answer_type"]
    c = _clean(str(pred))
    if t == "yesno":
        first = c.lower().split()[0] if c.split() else ""
        return YN.get(first, first)
    if t == "number":
        n = as_number(c)
        return str(n) if n is not None else "nan"
    return c


def _records(kwargs, n):
    keys = ("id", "gold", "answer_type", "system_answerable", "raw_prompt")
    return [{"id": kwargs["id"][i], "gold": kwargs["gold"][i], "answer_type": kwargs["answer_type"][i],
             "system_answerable": kwargs["system_answerable"][i], "prompt": kwargs["raw_prompt"][i],
             "sentences_json": (kwargs.get("sentences_json") or [""] * n)[i]}
            for i in range(n)] if all(k in kwargs for k in keys) else None


def _text(c) -> str:
    return c if isinstance(c, str) else c[0]["content"]


def make_reward(name: str):
    def fn(prompts, completions, **kwargs):
        recs = _records(kwargs, len(completions))
        out = []
        for rec, c in zip(recs, completions):
            comp = components(rec, _text(c))
            if name == "correct_x_valid":
                out.append(comp["correct"] * comp["valid"])
            elif name == "cvf":  # G7: correct x valid x faithful premises (bogus_mp is valid on 69% of items)
                out.append(comp["correct"] * comp["valid"] * comp["prem_ok"])
            elif name == "cvf_fmt":
                out.append(comp["correct"] * comp["valid"] * comp["prem_ok"] * comp["format_ok"])
            elif name == "gvc":
                out.append((comp["grammatical"] + comp["valid"] + comp["correct"]) / 3.0)
            else:
                out.append(comp[name])
        return out
    fn.__name__ = name
    return fn


ARMS = {"correct": "correct", "correct_x_valid": "correct_x_valid", "gvc": "gvc", "valid": "valid",
        "lines": "lines", "lines_fmt": "lines_fmt", "lines_raw": "lines_raw",
        "frac": "frac", "frac_hard": "frac_hard", "cvf": "cvf", "cvf_fmt": "cvf_fmt"}
LOGGED = ["correct", "valid", "grammatical", "in_system", "has_proof", "valid_strict", "lines", "n_parsed", "n_ok",
          "n_steps", "circular", "n_taut", "format_ok", "valid_prem", "frac_parsed", "frac_ok", "n_prem_bad", "prem_ok", "frac",
          "cvf", "cvf_fmt"]
# per-domain copies of the headline metrics (2026-09-30, L1 long study): the step mean mixes generator and Dolci
# prompts, whose rates differ by ~20x; these return None off-domain, which TRL logs as a nanmean over the domain
BY_BENCH = ["correct", "valid", "cvf", "has_proof", "grammatical"]
BENCH_GROUPS = {"gen": lambda b: b == "gen", "dolci": lambda b: str(b).startswith("dolci")}


def make_bench_reward(name: str, group: str):
    base, keep = make_reward(name), BENCH_GROUPS[group]

    def fn(prompts, completions, **kwargs):
        benches = kwargs.get("bench") or [None] * len(completions)
        vals = base(prompts, completions, **kwargs)
        return [v if keep(b) else None for v, b in zip(vals, benches)]
    fn.__name__ = f"{name}_{group}"
    return fn


def reward_funcs(arm: str):
    """[primary, *logged components, *per-domain components] and weights [1, 0, ...]."""
    primary = ARMS[arm]
    funcs = [make_reward(n) for n in [primary] + [n for n in LOGGED if n != primary]]
    funcs += [make_bench_reward(n, g) for g in BENCH_GROUPS for n in BY_BENCH]
    return funcs, [1.0] + [0.0] * (len(funcs) - 1)
