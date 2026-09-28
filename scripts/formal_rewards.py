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
  in_system    valid AND the proof's `ans` equals the reference
Arms (the primary reward; the other components are logged with weight 0):
  G0/G1 correct            G2 correct_x_valid        G3 gvc = (grammatical + valid + correct) / 3
  G4 valid
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_formal_bench_vllm import match, score  # noqa: E402

_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = 65536


def components(rec: dict, completion: str) -> dict:
    key = (rec["id"], completion)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        row = score(rec, completion)
    except Exception as e:  # a checker crash must not kill the run; count it as an invalid proof
        row = {"correct": False, "grammatical": False, "valid": False, "grounded": False, "sys_answer": None,
               "pred_answer": None, "has_proof": False, "error": f"crash: {e!r}"}
    agree = True
    if row.get("pred_answer") is not None and row.get("sys_answer") is not None:
        agree = match(str(row["sys_answer"]), {**rec, "gold": _gold_like(rec, row["pred_answer"])}) == 1.0
    valid = bool(row["valid"] and row["grounded"] and row.get("sys_answer") is not None and agree)
    in_sys = bool(valid and rec.get("system_answerable")
                  and match(str(row["sys_answer"]), rec) == 1.0)
    out = {"correct": float(bool(row["correct"])), "grammatical": float(bool(row["grammatical"])),
           "valid": float(valid), "in_system": float(in_sys), "has_proof": float(bool(row["has_proof"])),
           "valid_strict": float(bool(row["valid"]))}
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


ARMS = {"correct": "correct", "correct_x_valid": "correct_x_valid", "gvc": "gvc", "valid": "valid"}
LOGGED = ["correct", "valid", "grammatical", "in_system", "has_proof", "valid_strict"]


def reward_funcs(arm: str):
    """[primary, *logged components] and weights [1, 0, ...]."""
    primary = ARMS[arm]
    names = [primary] + [n for n in LOGGED if n != primary]
    return [make_reward(n) for n in names], [1.0] + [0.0] * (len(names) - 1)
