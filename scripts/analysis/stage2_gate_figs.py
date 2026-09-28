#!/usr/bin/env python
"""Figures + tables for reports/2026-09-28_stage2_gate.md (greedy gate on the Dolci RL sample)."""
import collections, json, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925")
FIG = Path(__file__).resolve().parents[2] / "reports/figures"
MET = ["has_proof", "grammatical", "valid", "correct", "in_system"]
BENCH = ["dolci_math", "dolci_dapo", "dolci_wordprob", "dolci_yesno", "dolci_knowledge"]


def err_type(r):
    if not r["has_proof"]:
        return "no proof"
    e = r.get("error") or {}
    code, msg = e.get("code", ""), e.get("msg", "")
    if not r["grammatical"]:
        return "parse: goal line" if re.search(r'at "goal', msg) or msg.startswith("syntax error at \"goal") else "parse: body"
    if code == "quote":
        return "quote not in prompt"
    if not r["valid"]:
        return "rule error" if code == "rule" else "other check error"
    if not r["grounded"]:
        return "ungrounded"
    return "valid"


runs = {}
for d in sorted(ROOT.glob("qwen35_*_p*_lr5em6_seed3407/rl_gate_dolci/generations.jsonl")):
    m = re.search(r"qwen35_([\d.]+b)_dolci_rlvlgen_p(\d+)", str(d))
    runs[f"{m[1]} p{int(m[2])}"] = [json.loads(l) for l in open(d)]
names = sorted(runs, key=lambda k: (float(k.split("b")[0]), int(k.split("p")[1])))
FIG.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, len(BENCH), figsize=(4 * len(BENCH), 3.6), sharey=True)
w = 0.8 / len(names)
for ax, b in zip(axes, BENCH):
    for j, n in enumerate(names):
        rs = [r for r in runs[n] if r["bench"] == b]
        ax.bar(np.arange(len(MET)) + j * w, [np.mean([r[m] for r in rs]) for m in MET], w, label=n)
    ax.set_xticks(np.arange(len(MET)) + 0.4 - w / 2, MET, rotation=35, ha="right")
    ax.set_title(b.replace("dolci_", ""))
axes[0].set_ylabel("rate (greedy)"); axes[-1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "stage2_gate_per_bench.png", dpi=130)

cats = ["no proof", "parse: goal line", "parse: body", "quote not in prompt", "rule error", "other check error", "ungrounded", "valid"]
fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(names)))
left = np.zeros(len(names))
for c in cats:
    v = np.array([collections.Counter(err_type(r) for r in runs[n])[c] / len(runs[n]) for n in names])
    ax.barh(names, v, left=left, label=c); left += v
ax.set_xlabel("fraction of 950 gate items"); ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35))
fig.tight_layout(); fig.savefig(FIG / "stage2_gate_errors.png", dpi=130)

print("| model | " + " | ".join(MET) + " | truncated* |")
print("|---" * (len(MET) + 2) + "|")
for n in names:
    rs = runs[n]
    print(f"| {n} | " + " | ".join(f"{np.mean([r[m] for r in rs]):.3f}" for m in MET)
          + f" | {np.mean([r['has_proof'] and 'Answer' not in r['generation'] and '</answer>' not in r['generation'] for r in rs]):.2f} |")
