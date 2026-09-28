#!/usr/bin/env python
"""In-domain rates vs mixture share X for every Stage-1 cell with a finished formal eval (2026-09-28).

Reads rlvl_data/formal_mixture_sft_20260925/qwen35_<size>_dolci_rlvlgen_p<X>_lr5em6_seed3407/formal_eval/summary.json
(2000 unseen generator problems per cell) and writes reports/figures/stage1_indomain_curves.png plus a
markdown table on stdout.
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

res = {}
for f in ROOT.glob("qwen35_*_dolci_rlvlgen_p*_lr5em6_seed3407/formal_eval/summary.json"):
    m = re.search(r"qwen35_([\d.]+b)_dolci_rlvlgen_p(\d+)", str(f))
    res.setdefault(m[1], {})[int(m[2])] = json.load(open(f))["overall"]
print("| model | X | " + " | ".join(l for _, l in MET) + " |\n|---|---|" + "---|" * len(MET))
for size in sorted(res, key=lambda s: float(s[:-1])):
    for x in sorted(res[size]):
        print(f"| {size.upper()} | {x} | " + " | ".join(f"{res[size][x][k]:.3f}".replace("0.", ".", 1) for k, _ in MET) + " |")
fig, axes = plt.subplots(1, len(MET), figsize=(4 * len(MET), 3.5), sharey=True)
for ax, (k, lbl) in zip(axes, MET):
    for size in sorted(res, key=lambda s: float(s[:-1])):
        xs = sorted(res[size])
        # sizes with few cells: points only, a line through 0 and 50 would suggest a measured trend
        ax.plot(xs, [res[size][x][k] for x in xs], "-o" if len(xs) >= 4 else "o", ms=4 if len(xs) >= 4 else 7,
                color=COL.get(size), label=size.upper())
    ax.set_title(lbl)
    ax.set_xlabel("X (% formal in SFT mix)")
    ax.grid(alpha=.3)
axes[0].set_ylabel("rate on 2000 unseen generator problems")
axes[0].legend()
fig.tight_layout()
fig.savefig(FIG / "stage1_indomain_curves.png", dpi=150)
