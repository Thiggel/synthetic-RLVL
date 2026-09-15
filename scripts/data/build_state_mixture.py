#!/usr/bin/env python3
"""Mix the chain-propagation corpora into the instruction mixture.

Same replacement design as the deduction sweep, so every condition trains on
the same number of examples and only the content of the replaced slice differs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolci", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True, help="jsonl from gen_state_tasks.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", choices=["formal", "nl"], default="formal")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--kinds", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    dolci = load_from_disk(str(args.dolci))
    k = int(round(args.frac * args.total))
    if len(dolci["train"]) < args.total:
        raise SystemExit(f"instruction corpus holds {len(dolci['train'])} < {args.total}")

    pool = [json.loads(l) for l in args.corpus.open(encoding="utf-8")]
    if args.kinds:
        pool = [r for r in pool if r["kind"] in set(args.kinds)]
    if len(pool) < k:
        raise SystemExit(f"corpus holds {len(pool)} items < {k} needed")
    rng = random.Random(args.seed)
    chosen = rng.sample(pool, k)
    traces = [{"prompt": r["prompt"], "target": r[args.template]} for r in chosen]

    kept = dolci["train"].select(range(args.total - k))
    rows = [{"prompt": p, "target": t} for p, t in zip(kept["prompt"], kept["target"])] + traces
    rng.shuffle(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows), "eval": dolci["eval"]}).save_to_disk(str(args.out))
    counts: dict[str, int] = {}
    for r in chosen:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    meta = dict(corpus=str(args.corpus), template=args.template, frac=args.frac,
                total=args.total, n_traces=k, n_instruction=args.total - k,
                kind_counts=counts, seed=args.seed,
                design="replacement (constant example count); eval split is instruction-only")
    (args.out / "mixture_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
