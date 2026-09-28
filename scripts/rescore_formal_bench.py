#!/usr/bin/env python
"""Re-score the answer fields of finished eval_formal_bench_vllm.py runs after a change
to answer extraction (2026-09-28: loose "**Answer:** x" / "the answer is x" fallback).

Only the answer-dependent fields (score/f1, correct, valid_correct, valid_wrong) are
recomputed from the stored generation, sys_answer and pred_answer; the checker fields
are kept (they depend on the possibly middle-cut prompt used at generation time).
The previous summary is kept as summary.pre_rescore.json.

  PYTHONPATH=RLVL-next/rlvl/python:RLVL-next/gen python scripts/rescore_formal_bench.py DIR [DIR ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_formal_bench_vllm import ANSWER_TAG_RE, BOXED_RE, LOOSE_ANSWER_RE, aggregate, match  # noqa: E402


def rescore(r):
    g = r["generation"]
    cand = r["pred_answer"] if r["pred_answer"] is not None else (
        str(r["sys_answer"]) if r["sys_answer"] is not None else None)
    if cand is None:
        tags, boxed = ANSWER_TAG_RE.findall(g), BOXED_RE.findall(g)
        cand = tags[-1] if tags else (boxed[-1] if boxed else None)
    if cand is None:
        nl = LOOSE_ANSWER_RE.findall(g)
        cand = nl[-1] if nl else None
    s = match(cand, r)
    r["f1" if r["answer_type"] == "span" else "score"] = s
    r["correct"] = s >= 0.5
    r["valid_correct"] = r["valid"] and r["correct"]
    r["valid_wrong"] = r["valid"] and not r["correct"]
    return r


for d in map(Path, sys.argv[1:]):
    gp, sp = d / "generations.jsonl", d / "summary.json"
    if not (gp.exists() and sp.exists()):
        continue
    rows = [rescore(json.loads(l)) for l in open(gp)]
    old = json.loads(sp.read_text())
    if not (d / "summary.pre_rescore.json").exists():
        (d / "summary.pre_rescore.json").write_text(json.dumps(old, indent=2) + "\n")
    new = {**{k: old[k] for k in old if k not in ("overall", "per_group", "per_bench")}, "rescored": "2026-09-28 loose answer",
           **aggregate(rows)}
    gp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    sp.write_text(json.dumps(new, indent=2) + "\n")
    print(f"{d}: correct {old['overall']['all']['correct']:.3f} -> {new['overall']['all']['correct']:.3f}")
