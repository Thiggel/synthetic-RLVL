#!/usr/bin/env python3
"""Self-play prompt set for AlphaZero round 1 (2026-10-04).

RL train prompts (gate near-duplicates excluded), in the gate test.jsonl schema that scripts/az/mcts_decode.py reads.
Difficulty comes from the EI-4 harvest (rl_filter_20261003, teacher G16@750, cvf at n=16): per bench we take the
`mixed` prompts (0 < rate < 1, where sampling already finds a verified correct proof sometimes) and `zero` prompts
(rate 0: the case where search must beat sampling). `full` prompts teach nothing. Writes --shards jsonl files.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_formal import build_dataset  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
FILT = DATA / "rl_filter_20261003"
CONTAM = DATA / "datasets/rl_gate_dolci_instruct_20260928/contamination.json"
QUOTA = {"gsm8k_train": (600, 400), "dolci_wordprob": (265, 235), "dolci_math": (277, 223)}  # (mixed, zero)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "az/selfplay_r1")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every mixed and zero prompt, no per-bench quota (online AZ pool)")
    args = ap.parse_args()
    rates = {}
    for f in ("G16c750_cvf_n16_gsm8k.json", "G16c750_cvf_n16_dolci.json"):
        rates.update(json.load(open(FILT / f))["rates"])
    exclude = set(json.load(open(CONTAM))["pool_exclude_ids"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DATA / "formal_mixture_sft_20260925/qwen35_2b_lc_libext_e4_lr5em6_seed3407/final")
    ds = build_dataset(tok, True, None, args.seed, list(QUOTA), exclude_ids=exclude)
    rng = random.Random(args.seed)
    pools = {(b, k): [] for b in QUOTA for k in ("mixed", "zero")}
    for r in ds:
        x = rates.get(r["id"])
        if x is None:
            continue
        k = "zero" if x["rate"] == 0 else "mixed" if x["rate"] < 1 else None
        if k:
            pools[(r["bench"], k)].append({"bench": r["bench"], "group": "rl_train", "id": r["id"],
                                           "prompt": r["raw_prompt"], "gold": r["gold"], "answer_type": r["answer_type"],
                                           "gold_label": r["gold"], "system_answerable": r["system_answerable"],
                                           "teacher_rate": x["rate"], "difficulty": k})
    rows = []
    for b, (nm, nz) in QUOTA.items():
        for k, n in (("mixed", nm), ("zero", nz)):
            p = pools[(b, k)]
            n = len(p) if args.all else n
            rng.shuffle(p)
            rows += p[:n]
            print(b, k, f"{min(n, len(p))}/{len(p)}")
    rng.shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for s in range(args.shards):
        with open(args.out / f"prompts_{s}.jsonl", "w") as f:
            for r in rows[s::args.shards]:
                f.write(json.dumps(r) + "\n")
    print(len(rows), "prompts ->", args.out)


if __name__ == "__main__":
    main()
