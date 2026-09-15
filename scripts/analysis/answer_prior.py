#!/usr/bin/env python3
"""Measure a model's answer distribution against the benchmark's label distribution.

The gain from replacing part of an instruction mixture with generated
derivations depends on the prior the model starts with. Qwen2.5-7B answers
"true" far more often than the labels warrant and gains when the derivations
pull it back. Llama-3.1-8B leans toward "unknown" and loses when the same data
pushes it further. This script reports both distributions from saved samples so
the diagnosis can be run before choosing what to inject.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re


def first_word(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "<empty>"
    word = re.split(r"[\s<]+", text)[0]
    return word.lower().strip(".,:;")


def read(run_dir: str, pattern: str) -> tuple[collections.Counter, collections.Counter, float, int]:
    pred, gold, correct, n = collections.Counter(), collections.Counter(), 0.0, 0
    for path in glob.glob(os.path.join(run_dir, "**", pattern), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                resp = row.get("filtered_resps") or row.get("resps")
                while isinstance(resp, list) and resp:
                    resp = resp[0]
                pred[first_word(resp)] += 1
                target = row["doc"].get("answer") or row.get("target")
                gold[str(target).strip().lower()] += 1
                correct += float(row.get("exact_match") or row.get("acc") or 0.0)
                n += 1
    return pred, gold, correct, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="suite directory holding run directories")
    ap.add_argument("--pattern", default="samples_*.jsonl*")
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    runs = args.runs or sorted(
        r for r in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, r)) and os.path.exists(os.path.join(args.root, r, ".complete"))
    )
    for run in runs:
        pred, gold, correct, n = read(os.path.join(args.root, run), args.pattern)
        if not n:
            continue
        labels = sorted(gold, key=lambda k: -gold[k])[:4]
        print(f"{run}  n={n}  accuracy={100 * correct / n:.1f}")
        print("   gold      " + "  ".join(f"{k}={gold[k] / n:.3f}" for k in labels))
        print("   predicted " + "  ".join(f"{k}={pred.get(k, 0) / n:.3f}" for k in labels))
        excess = max(labels, key=lambda k: pred.get(k, 0) / n - gold[k] / n)
        print(f"   largest excess: {excess} by {100 * (pred.get(excess, 0) - gold[excess]) / n:+.1f} points")


if __name__ == "__main__":
    main()
