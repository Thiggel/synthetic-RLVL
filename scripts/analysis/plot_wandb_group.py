from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


def collect_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    return api.runs(f"{entity}/{project}", filters={"group": group})


def aggregate_metric(runs, metric: str):
    series = []
    for run in runs:
        hist = run.history(keys=["_step", metric], pandas=True)
        if metric not in hist:
            continue
        vals = hist[["_step", metric]].dropna()
        if len(vals) == 0:
            continue
        series.append(vals)
    if not series:
        return None, None, None

    all_steps = sorted(set(int(s) for df in series for s in df["_step"].tolist()))
    mat = np.full((len(series), len(all_steps)), np.nan)
    for i, df in enumerate(series):
        lookup = {int(r["_step"]): float(r[metric]) for _, r in df.iterrows()}
        for j, s in enumerate(all_steps):
            if s in lookup:
                mat[i, j] = lookup[s]
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return np.array(all_steps), mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--step-min", type=int, default=1)
    ap.add_argument("--step-max", type=int, default=20)
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    runs = collect_runs(args.entity, args.project, args.group)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, label in [("format", "Format"), ("correct", "Correctness"), ("valid", "Validity")]:
        plt.figure(figsize=(10, 6))
        for step in range(args.step_min, args.step_max + 1):
            metric = f"gen/step_{step}/{metric_name}"
            xs, mean, std = aggregate_metric(runs, metric)
            if xs is None:
                continue
            plt.plot(xs, mean, label=f"n={step}")
            plt.fill_between(xs, mean - std, mean + std, alpha=0.15)

        plt.xlabel("train step")
        plt.ylabel(label)
        plt.title(f"{args.group}: {label} by reasoning depth")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{args.group}_{metric_name}.pdf")
        plt.close()


if __name__ == "__main__":
    main()
