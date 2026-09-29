#!/usr/bin/env python
"""In-domain rates vs the number of generator rows seen in SFT (2026-09-29).

Question (user, 2026-09-29): would a longer SFT (e.g. 10x) give a better base? Two sources:
  mixture sweep   qwen35_<size>_dolci_rlvlgen_p<X>: one epoch over 100k rows, X% of them generator
                  rows, so X * 1000 generator rows (the Dolci share shrinks as X grows)
  scaled p50      qwen35_2b_dolci_rlvlgen_p50_x3: p50 at 3x the rows (300k, 2344 steps); its
                  checkpoints at steps 781 / 1562 / final have seen ~50k / 100k / 150k generator rows
                  (rows are shuffled), always at a 50% share
Reads formal_eval/summary.json (2000 unseen generator problems) of every run / checkpoint present
and writes reports/figures/sft_scale_curve.png.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925")
FIG = Path(__file__).resolve().parents[2] / "reports/figures"
MET = [("faithful", "faithful"), ("grammatical", "grammatical"), ("valid", "valid"), ("answer_acc", "answer acc")]
COL = {"0.8b": "#4C78A8", "2b": "#F58518", "9b": "#54A24B"}
SCALED = {"2b": ROOT / "qwen35_2b_dolci_rlvlgen_p50_x3_lr5em6_seed3407"}
STEPS_PER_50K = 781

sweep, scaled = {}, {}
for f in ROOT.glob("qwen35_*_dolci_rlvlgen_p*_lr5em6_seed3407/formal_eval/summary.json"):
    m = re.search(r"qwen35_([\d.]+b)_dolci_rlvlgen_p(\d+)_lr", str(f))
    if m and int(m[2]) > 0:
        sweep.setdefault(m[1], {})[int(m[2]) * 1000] = json.load(open(f))["overall"]
for size, run in SCALED.items():
    for f in run.glob("*/formal_eval/summary.json"):
        ck = f.parts[-3]
        step = int(ck.split("-")[1]) if ck.startswith("checkpoint-") else 3 * STEPS_PER_50K
        scaled.setdefault(size, {})[round(step / STEPS_PER_50K) * 50_000] = json.load(open(f))["overall"]

fig, axes = plt.subplots(1, len(MET), figsize=(4.2 * len(MET), 3.7), sharey=True)
for ax, (k, lbl) in zip(axes, MET):
    for size in sorted(sweep, key=lambda s: float(s[:-1])):
        xs = sorted(sweep[size])
        ax.plot(xs, [sweep[size][x][k] for x in xs], "-o", ms=4, color=COL.get(size),
                label=f"{size.upper()} sweep (100k rows, X% formal)")
    for size, pts in scaled.items():
        xs = sorted(pts)
        ax.plot(xs, [pts[x][k] for x in xs], "--s", ms=6, color=COL.get(size), mfc="white",
                label=f"{size.upper()} p50 x3 (300k rows) checkpoints")
    ax.set_xscale("log", base=2)
    ax.set_xticks([5e3, 1e4, 2.5e4, 5e4, 1e5, 1.5e5], ["5k", "10k", "25k", "50k", "100k", "150k"])
    ax.set_title(lbl)
    ax.set_xlabel("generator rows seen in SFT")
    ax.grid(alpha=.3)
axes[0].set_ylabel("rate on 2000 unseen generator problems")
axes[0].legend(fontsize=7)
fig.suptitle("Does more SFT help? In-domain rates vs generator rows seen")
fig.tight_layout()
fig.savefig(FIG / "sft_scale_curve.png", dpi=140)
for size in sweep:
    print(size, {x: round(v["valid"], 3) for x, v in sorted(sweep[size].items())})
for size in scaled:
    print(size, "x3", {x: round(v["valid"], 3) for x, v in sorted(scaled[size].items())})
