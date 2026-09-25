#!/usr/bin/env python
"""Aggregate the 2026-09-25 formal-CoT mixture sweep evals.

Reads <out-root>/qwen35_{model}_dolci_rlvlgen_pXX_lr5em6_seed3407/formal_eval/summary.json
(written by scripts/eval_formal_vllm.py) and the trainer's final_eval_metrics.json,
and writes to --out-dir:
  results_overall.csv     one row per (model, X)
  results_per_family.csv  one row per (model, X, family)
  curves_overall.{pdf,png}     metric vs X, one line per model
  curves_per_family.{pdf,png}  metric vs X per family (grid), one line per model
Missing cells are listed and skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

METRICS = ["faithful", "grammatical", "valid", "answer_acc", "answer_acc_lenient",
           "given_precision", "given_recall"]
PLOT_METRICS = ["faithful", "grammatical", "valid", "answer_acc", "answer_acc_lenient"]
MODELS = ["0.8b", "2b", "9b"]
RUN_RE = re.compile(r"qwen35_(?P<m>[0-9.]+b)_dolci_rlvlgen_p(?P<x>\d\d)_lr5em6_seed3407$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/formal_mixture_sweep_20260925/results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall, fam_rows, missing = [], [], []
    for d in sorted(args.out_root.iterdir()):
        m = RUN_RE.match(d.name)
        if not m:
            continue
        model, x = m["m"], int(m["x"])
        s = d / "formal_eval" / "summary.json"
        if not s.exists():
            missing.append(d.name)
            continue
        summ = json.loads(s.read_text())
        fe = d / "final" / "final_eval_metrics.json"
        eval_loss = json.loads(fe.read_text()).get("eval_loss") if fe.exists() else None
        overall.append({"model": model, "x": x, "n": summ["overall"]["n"], "dolci_eval_loss": eval_loss,
                        **{k: summ["overall"].get(k) for k in METRICS}})
        for fam, v in sorted(summ["per_family"].items()):
            fam_rows.append({"model": model, "x": x, "family": fam, "n": v["n"], **{k: v.get(k) for k in METRICS}})

    overall.sort(key=lambda r: (MODELS.index(r["model"]), r["x"]))
    fam_rows.sort(key=lambda r: (MODELS.index(r["model"]), r["x"], r["family"]))
    for name, rows in [("results_overall.csv", overall), ("results_per_family.csv", fam_rows)]:
        if rows:
            with open(args.out_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"{len(overall)} cells; missing evals: {missing}")
    for r in overall:
        print(f"{r['model']:>4} p{r['x']:02d} " + " ".join(
            f"{k}={r[k]:.3f}" for k in PLOT_METRICS if r[k] is not None) +
            (f" dolci_loss={r['dolci_eval_loss']:.4f}" if r["dolci_eval_loss"] else ""))
    if not overall:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"0.8b": "#4C78A8", "2b": "#F58518", "9b": "#54A24B"}

    def lines(ax, rows, metric):
        for mdl in MODELS:
            rs = [r for r in rows if r["model"] == mdl and r[metric] is not None]
            if rs:
                ax.plot([r["x"] for r in rs], [r[metric] for r in rs], "o-", ms=3, lw=1.5,
                        color=colors[mdl], label=f"Qwen3.5-{mdl.upper()}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(3.2 * len(PLOT_METRICS), 3.0), sharey=True)
    for ax, met in zip(axes, PLOT_METRICS):
        lines(ax, overall, met)
        ax.set_title(met)
        ax.set_xlabel("% rlvlgen in SFT mix")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"curves_overall.{ext}", dpi=150)
    plt.close(fig)

    fams = sorted({r["family"] for r in fam_rows})
    fig, axes = plt.subplots(len(fams), len(PLOT_METRICS), figsize=(2.6 * len(PLOT_METRICS), 1.9 * len(fams)),
                             sharex=True, sharey=True, squeeze=False)
    for i, fam in enumerate(fams):
        rows = [r for r in fam_rows if r["family"] == fam]
        for j, met in enumerate(PLOT_METRICS):
            lines(axes[i][j], rows, met)
            if i == 0:
                axes[i][j].set_title(met, fontsize=9)
            if j == 0:
                axes[i][j].set_ylabel(fam, fontsize=9)
    axes[0][0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("% rlvlgen")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"curves_per_family.{ext}", dpi=120)
    plt.close(fig)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
