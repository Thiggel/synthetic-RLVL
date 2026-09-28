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
  lines_raw  the literal formula on the same line counts, (n_parsed + n_ok) * (1 + correct),
             for comparison only (rewards length)
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_formal_bench_vllm import match, score  # noqa: E402

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
    """
    import rlvl
    zero = {"n_steps": 0, "n_parsed": 0, "n_ok": 0, "circular": 0, "n_taut": 0}
    s = completion.find("<proof>\n")
    if s < 0:
        return zero
    e = completion.find("</proof>", s)
    body = completion[s + len("<proof>\n"): e if e >= 0 else len(completion)]
    cites, formula, rule, order, root = {}, {}, {}, [], None
    for ln in body.split("\n"):
        m = _ANS.match(ln)
        if m:
            root = m.group(1)
            break
        m = _STEP.match(ln)
        if m:
            lab, f, just = m.groups()
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
        stack += cites[x]
    try:
        rep = rlvl.check(prompt, body, strict=True)
    except BaseException:  # pyo3 PanicException is not an Exception
        return {**zero, "n_steps": len(order)}
    parsed, ok, taut = {}, {}, set()
    for r in rep.get("lines") or []:
        if r.get("kind") == "step" and r.get("label") in anc:
            f = formula.get(r["label"])
            if rule.get(r["label"]) != "given" and tautology(f):
                taut.add(f)
                continue
            parsed[f] = True
            ok[f] = ok.get(f, False) or bool(r.get("ok") and r.get("rule") != "given")
    norm = lambda f: "".join(f.split())
    circ = any(rule[x] == "given" and norm(formula[x]) == norm(formula[root]) for x in anc)
    return {"n_steps": len(order), "n_parsed": len(parsed), "n_ok": sum(ok.values()), "circular": int(circ),
            "n_taut": len(taut)}


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
    in_sys = bool(valid and rec.get("system_answerable")
                  and match(str(row["sys_answer"]), rec) == 1.0)
    out = {"correct": float(bool(row["correct"])), "grammatical": float(bool(row["grammatical"])),
           "valid": float(valid), "in_system": float(in_sys), "has_proof": float(bool(row["has_proof"])),
           "valid_strict": float(bool(row["valid"])), "format_ok": float(format_ok(completion)),
           **{k: float(v) for k, v in ls.items()}}
    g, v = min(1.0, ls["n_parsed"] / LINE_CAP), min(1.0, ls["n_ok"] / LINE_CAP)
    out["lines"] = ((g + v) / 2 + valid) * (1 + out["correct"]) / 4
    out["lines_fmt"] = out["lines"] * out["format_ok"]
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
             "system_answerable": kwargs["system_answerable"][i], "prompt": kwargs["raw_prompt"][i]}
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
            elif name == "gvc":
                out.append((comp["grammatical"] + comp["valid"] + comp["correct"]) / 3.0)
            else:
                out.append(comp[name])
        return out
    fn.__name__ = name
    return fn


ARMS = {"correct": "correct", "correct_x_valid": "correct_x_valid", "gvc": "gvc", "valid": "valid",
        "lines": "lines", "lines_fmt": "lines_fmt", "lines_raw": "lines_raw"}
LOGGED = ["correct", "valid", "grammatical", "in_system", "has_proof", "valid_strict", "lines", "n_parsed", "n_ok",
          "n_steps", "circular", "n_taut", "format_ok"]


def reward_funcs(arm: str):
    """[primary, *logged components] and weights [1, 0, ...]."""
    primary = ARMS[arm]
    names = [primary] + [n for n in LOGGED if n != primary]
    return [make_reward(n) for n in names], [1.0] + [0.0] * (len(names) - 1)
