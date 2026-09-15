#!/usr/bin/env python3
"""Build a training set of generated derivations alone.

The replacement design asks how much instruction data can be given up. This is
the endpoint of that question, with none of it left. The band 25 corpus holds
50,000 rows, so half the examples come from the band 15 corpus, which makes the
depth distribution shallower than the five percent arm and is stated in the
manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_reasoning_mixture_sft import render_traces  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolci", type=Path, required=True, help="only its eval split is used")
    ap.add_argument("--bp-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", required=True, choices=["logic", "nl_exact"])
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    half = args.total // 2
    rows = render_traces(args.bp_root, 25, args.template, half, args.seed)
    rows += render_traces(args.bp_root, 15, args.template, args.total - half, args.seed + 1)
    random.Random(args.seed).shuffle(rows)

    dolci = load_from_disk(str(args.dolci))
    args.out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows), "eval": dolci["eval"]}).save_to_disk(str(args.out))
    meta = dict(template=args.template, total=args.total, n_traces=len(rows), n_instruction=0,
                bands={"25": half, "15": args.total - half}, seed=args.seed,
                design="no instruction data; eval split is instruction-only for comparability")
    (args.out / "mixture_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
