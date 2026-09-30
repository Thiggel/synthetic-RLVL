#!/usr/bin/env python
"""Stage-2 gate set: a fixed held-out sample of the Olmo 3 RL prompts (allenai/Dolci-Instruct-RL).

docs/research_plan.md requires that the Stage-1 policy reach a tagged in_system rate
clearly above 0 on the RL prompt distribution before GRPO with validity rewards
(otherwise the reward has no signal). This writes the sample in the schema of
build_formal_bench_tagged.py, so scripts/eval_formal_bench_vllm.py --test-jsonl
scores it unchanged, plus the list of held-out ids that the RL training set must exclude.

Checkable subsets (answer type from ground_truth):
  dolci_math        math (numeric / p/q answer), OpenMath-style generated problems
  dolci_dapo        math_dapo (competition, integer answer)
  dolci_wordprob    tulu_3_rewritten persona word problems with a numeric answer
  dolci_yesno       tulu_3_rewritten / math items with a yes/no answer (the "ambiguous" type)
  dolci_knowledge   virtuoussy multi-subject short answers (no premises in the prompt: not
                    system-answerable, exact-match only; tracks what the system cannot express)
Excluded from Stage 2 (no checkable answer): code, ifeval, general-quality, long free-text refs.

  HF_HOME=/vol/tmp2/laitenbf HF_HUB_OFFLINE=1 .venv_rlvl_vllm/bin/python scripts/build_rl_gate_set.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
from fractions import Fraction
from pathlib import Path

OUT = Path(os.environ.get("RLVL_DATA_ROOT", "/vol/tmp2/laitenbf/rlvl_data")) / "datasets/rl_gate_dolci_instruct_20260928"
YN = {"yes": "yes", "no": "no", "true": "yes", "false": "no"}


def gt_of(r) -> str:
    g = r["ground_truth"]
    return str(g[0] if isinstance(g, list) else g).strip()


def as_fraction(g: str):
    g = g.replace(",", "")
    if not re.fullmatch(r"-?\d+(\.\d+)?(/\d+)?", g):
        return None
    try:
        return Fraction(g)
    except (ValueError, ZeroDivisionError):
        return None


def classify(r):
    """(bench, answer_type, gold, system_answerable) or None if not checkable."""
    src, orig = r["dataset"][0], (r["original_dataset"] or r["data_source"] or "")
    g = gt_of(r)
    num = as_fraction(g)
    if src == "math":
        if g.lower() in YN:
            return "dolci_yesno", "yesno", YN[g.lower()], True
        if num is None:
            return None
        return ("dolci_dapo" if orig == "math_dapo" else "dolci_math"), "number", str(num), True
    if src != "general-quality_ref":
        return None
    if "tulu_3_rewritten" in orig:
        if g.lower() in YN:
            return "dolci_yesno", "yesno", YN[g.lower()], True
        if num is not None:
            return "dolci_wordprob", "number", str(num), True
        return None
    if "virtuoussy" in orig and len(g.split()) <= 5:
        return "dolci_knowledge", "text", g, False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--per-bench", default="dolci_math=300,dolci_dapo=150,dolci_wordprob=200,dolci_yesno=200,"
                                           "dolci_knowledge=100")
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("allenai/Dolci-Instruct-RL", split="train")
    pools, excluded = collections.defaultdict(list), collections.Counter()
    for i, r in enumerate(ds):
        p = r["prompt"]
        if not p.startswith("user: ") or "\nassistant:" in p:
            excluded["multi_turn"] += 1
            continue
        c = classify(r)
        if c is None:
            excluded[r["dataset"][0]] += 1
            continue
        pools[c[0]].append((i, r, c))
    rng = random.Random(args.seed)
    rows = []
    for spec in args.per_bench.split(","):
        bench, n = spec.split("=")
        for i, r, (b, t, g, a) in sorted(rng.sample(pools[bench], min(int(n), len(pools[bench]))), key=lambda x: x[0]):
            rows.append({"bench": b, "group": "dolci_rl", "id": f"{b}/{i}", "row": i,
                         "prompt": r["prompt"].removeprefix("user: ").strip(), "gold": g, "answer_type": t,
                         "gold_label": gt_of(r), "system_answerable": a})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "test.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (args.out_dir / "heldout_rows.json").write_text(json.dumps(sorted(r["row"] for r in rows)))
    stats = {"n_total": len(ds), "pool": {k: len(v) for k, v in pools.items()}, "excluded": dict(excluded),
             "sample": dict(collections.Counter(r["bench"] for r in rows))}
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
