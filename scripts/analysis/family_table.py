"""Summarise the family and scale study into the table the paper reports.

Reads an lm-eval results tree laid out as <root>/<suite>/<run>/**/results_*.json
and prints one row per base model with the control, the five percent formal
condition and the ten percent English condition side by side.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

CONDITIONS = ["control", "logic_p05", "nl_exact_p10"]
RUN_RE = re.compile(r"(?P<model>.+?)_(?P<cond>control|logic_p05|nl_exact_p10)_seed(?P<seed>\d+)$")


def metric_of(entry):
    for key, value in entry.items():
        if key == "alias" or "stderr" in key:
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def read_suite(root, suite):
    out = defaultdict(dict)
    directory = os.path.join(root, suite)
    if not os.path.isdir(directory):
        return out
    for run in sorted(os.listdir(directory)):
        run_dir = os.path.join(directory, run)
        if not os.path.exists(os.path.join(run_dir, ".complete")):
            continue
        files = sorted(glob.glob(os.path.join(run_dir, "**", "results_*.json"), recursive=True))
        if not files:
            continue
        results = json.load(open(files[-1]))["results"]
        match = RUN_RE.match(run)
        if not match:
            continue
        key = (match.group("model"), match.group("cond"))
        for task, entry in results.items():
            out[key].setdefault(task, []).append(metric_of(entry))
    return out


def mean(values):
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--suite", default="deduction")
    ap.add_argument("--task", default="synthrlvl_deduction_pw_d3")
    args = ap.parse_args()
    data = read_suite(args.root, args.suite)
    models = sorted({model for model, _ in data})
    print(f"{'model':16s} " + " ".join(f"{c:>12s}" for c in CONDITIONS))
    for model in models:
        cells = []
        for cond in CONDITIONS:
            values = data.get((model, cond), {}).get(args.task, [])
            cells.append(f"{100 * mean(values):12.1f}" if values else f"{'-':>12s}")
        print(f"{model:16s} " + " ".join(cells))


if __name__ == "__main__":
    main()
