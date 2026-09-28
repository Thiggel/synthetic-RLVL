"""Stage-1 report figures: in-system skill vs. general-capability cost per mix %.

Reads the in-domain results (results_overall.csv) and the untagged downstream
suite (bench_long.csv) of the 2026-09-25 formal mixture sweep and writes
reports/figures/stage1_*.{png,pdf}.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "analysis/formal_mixture_sweep_20260925"
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"0.8b": "#4C78A8", "2b": "#F58518", "9b": "#54A24B"}
GENERAL = ["gsm8k", "mmlu", "arc_challenge", "BBH (all)", "hellaswag", "winogrande", "humaneval", "mbpp"]
REASON = ["PW CoT d3", "PW CoT d5", "FOLIO", "BBH logical_deduction_five_objects", "GPQA-Diamond",
          "HotpotQA", "2WikiMQA", "MuSiQue"]

indomain = defaultdict(dict)
for r in csv.DictReader(open(A / "results/results_overall.csv")):
    indomain[r["model"]][int(r["x"])] = {k: float(v) for k, v in r.items() if k in
                                         ("faithful", "grammatical", "valid", "answer_acc")}
bench = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(A / "bench/bench_long.csv")):
    bench[r["model"]][int(r["x"])][r["benchmark"]] = float(r["score"])


def avg_delta(m, x, names):
    b0, bx = bench[m][0], bench[m][x]
    ds = [bx[n] - b0[n] for n in names if n in bx and n in b0]
    return sum(ds) / len(ds) if ds else None


# 1) trade-off: in-domain valid rate vs. mean delta on general benchmarks
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for m in sorted(indomain):
    xs = sorted(x for x in indomain[m] if x in bench[m] and x > 0)
    for j, (names, title) in enumerate([(GENERAL, "general (GSM8K, MMLU, ARC, BBH, HS, WG, code)"),
                                        (REASON, "reasoning (PW CoT, FOLIO, BBH-LD5, GPQA, multihop)")]):
        pts = [(avg_delta(m, x, names), indomain[m][x]["valid"], x) for x in xs]
        pts = [p for p in pts if p[0] is not None]
        ax[j].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COL.get(m), label=f"Qwen3.5-{m.upper()}")
        for d, v, x in pts:
            ax[j].annotate(f"{x}%", (d, v), textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax[j].axvline(0, color="grey", lw=0.8, ls="--")
        ax[j].set_xlabel(f"mean Δ vs. 0% baseline [points]\n{title}")
        ax[j].set_ylabel("in-domain valid-proof rate")
        ax[j].grid(alpha=0.3)
ax[0].legend()
fig.suptitle("Stage 1: formal-reasoning skill vs. cost on untagged benchmarks, per mix %")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"stage1_tradeoff.{ext}", dpi=160)

# 2) heatmap of deltas per benchmark
names = [n for n in GENERAL + REASON + ["PW d3", "PW d5", "GPQA-quant", "agieval_logiqa_en", "piqa"]
         if any(n in bench[m][0] for m in bench)]
fig, axs = plt.subplots(1, len(bench), figsize=(5.5 * len(bench), 0.33 * len(names) + 1.5), squeeze=False)
for k, m in enumerate(sorted(bench)):
    xs = sorted(x for x in bench[m] if x > 0)
    M = [[(bench[m][x].get(n, float("nan")) - bench[m][0].get(n, float("nan"))) for x in xs] for n in names]
    a = axs[0][k]
    im = a.imshow(M, cmap="RdBu", vmin=-15, vmax=15, aspect="auto")
    a.set_xticks(range(len(xs)), [f"{x}%" for x in xs], fontsize=8)
    a.set_yticks(range(len(names)), names if k == 0 else [""] * len(names), fontsize=8)
    for i in range(len(names)):
        for j in range(len(xs)):
            v = M[i][j]
            if v == v:
                a.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6)
    a.set_title(f"Qwen3.5-{m.upper()}: Δ vs. 0% (untagged)")
fig.colorbar(im, ax=axs[0].tolist(), shrink=0.6, label="Δ points")
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"stage1_bench_delta_heatmap.{ext}", dpi=160, bbox_inches="tight")
print("wrote", FIG)
