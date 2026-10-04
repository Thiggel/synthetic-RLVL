#!/usr/bin/env python3
"""Self-play harvest -> EI rows for the az1 student (Stage 3 round 1, 2026-10-04).

Input: MCTS self-play dirs (scripts/az/mcts_decode.py --terminal gold, generations.jsonl) on the round-1 prompt sets
(scripts/az/build_selfplay_prompts.py, prompts_*.jsonl, which hold the raw question under "prompt"). With a gold
terminal the search commits to answers it cannot verify at move-temp 1, so "found" proofs can still be wrong: keep only
rows with cvf = correct * valid_s2 * prem_ok = 1. Output: the rl_prompt_filter `<out>.passing.jsonl` schema
(id, bench, raw_prompt, gold, reward, completion) plus difficulty / teacher_rate, so
scripts/data/build_libext_ei_mixture.py can consume it like any EI harvest. Prints keep rates by bench x difficulty.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="self-play out dirs")
    ap.add_argument("--prompts", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    recs = {r["id"]: r for p in args.prompts for r in map(json.loads, open(p))}
    stats = collections.defaultdict(collections.Counter)
    n = 0
    with open(args.out, "w") as f:
        for d in args.runs:
            for g in map(json.loads, open(d / "generations.jsonl")):
                rec = recs[g["id"]]
                st = stats[(g["bench"], rec.get("difficulty", "?"))]
                st["items"] += 1
                st["found"] += bool(g["found"])
                if not (g["found"] and g["cvf"] == 1):
                    continue
                st["kept"] += 1
                n += 1
                f.write(json.dumps({"id": g["id"], "bench": g["bench"], "raw_prompt": rec["prompt"], "gold": g["gold"],
                                    "reward": 1.0, "completion": g["generation"], "difficulty": rec.get("difficulty"),
                                    "teacher_rate": rec.get("teacher_rate"), "source": str(d)}) + "\n")
    print(f"{n} rows -> {args.out}")
    for (b, diff), st in sorted(stats.items()):
        print(f"{b:16s} {diff:6s} items {st['items']:4d} found {st['found'] / st['items']:.3f} "
              f"kept(cvf) {st['kept'] / st['items']:.3f}")


if __name__ == "__main__":
    main()
