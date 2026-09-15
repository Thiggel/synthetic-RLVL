#!/usr/bin/env python3
"""Read RULER results and keep only the runs that built a long haystack.

This harness version names its metrics from a default list, so a results file
can report a score under the key 4096 when the items were built at 16,384 or
32,768 tokens. The length the run actually used is recorded in its own log, so
that is what decides whether a run counts.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics as st


def built_lengths(run_dir: str) -> set[int]:
    """Read the lengths from the saved samples, which record each item's own
    built length. A job log can belong to an earlier evaluation of the same
    run, so the samples are the only record that cannot be stale."""
    import gzip
    lengths: set[int] = set()
    for path in glob.glob(os.path.join(run_dir, "**", "samples_*.jsonl*"), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as handle:
            for line in handle:
                doc = json.loads(line).get("doc", {})
                if "max_length" in doc:
                    lengths.add(int(doc["max_length"]))
                break
    return lengths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-length", type=int, default=16384)
    args = ap.parse_args()
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    skipped = []
    for run in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, run)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, ".complete")):
            continue
        lengths = built_lengths(d)
        if not lengths or max(lengths) < args.min_length:
            skipped.append((run, sorted(lengths)))
            continue
        files = sorted(glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True))
        results = json.load(open(files[-1]))["results"]
        m = re.match(r"qwen25_7b_dose_(.+)_p(\d+)_seed(\d+)", run)
        cond = m.group(1) + m.group(2)
        for task, entry in results.items():
            for key, value in entry.items():
                if isinstance(value, float) and value != -1 and "stderr" not in key:
                    agg[cond][task].append(100 * value)
    tasks = sorted({t for c in agg for t in agg[c]})
    print(f"{'condition':14s}" + "".join(f"{t:>20s}" for t in tasks) + "  seeds")
    for cond in sorted(agg):
        n = len(agg[cond][tasks[0]]) if tasks else 0
        print(f"{cond:14s}" + "".join(f"{st.mean(agg[cond][t]):20.1f}" for t in tasks) + f"  {n}")
    if skipped:
        print("\nskipped, built only short contexts:")
        for run, lengths in skipped:
            print(f"  {run} {lengths}")


if __name__ == "__main__":
    main()
