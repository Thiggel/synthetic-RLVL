#!/usr/bin/env python3
"""Sampled pass@k on the Dolci RL gate (950 items, 16 samples at T=1.0, the GRPO rollout temperature).

Question (user, 2026-09-30): the greedy gate says more SFT data barely raises validity on Dolci; does it
raise has_proof@k / valid@k, i.e. how often GRPO with 8 rollouts sees a valid proof at all?
mixed@8 = fraction of prompts whose 8-rollout group has both passes and fails (non-zero GRPO advantage
under that metric as a binary reward). Inputs: <model>/rl_gate_dolci_k16/summary.json
(scripts/eval_formal_bench_vllm.py --n-samples 16 --temperature 1.0). Writes a markdown table and
reports/figures/passk_ladder.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
SFT = DATA / "formal_mixture_sft_20260925"
GRPO = DATA / "grpo_formal_20260928"
X3 = SFT / "qwen35_2b_dolci_rlvlgen_p50_x3_lr5em6_seed3407"
MODELS = [("SFT p25 fp32m (~25k gen rows)", SFT / "qwen35_2b_dolci_rlvlgen_p25_lr5em6_seed3407_fp32m/final"),
          ("SFT x3 @781 (~50k)", X3 / "checkpoint-781"),
          ("SFT x3 @1562 (~100k)", X3 / "checkpoint-1562"),
          ("SFT x3 final (~150k)", X3 / "final"),
          ("SFT p50 + lemma catalog fp32m", SFT / "qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407/final"),
          ("GRPO G8 final (lemma-catalog base, 200 steps)", GRPO / "2b_lcfp32m_G8_cvf/final")]
METRICS = ["has_proof", "grammatical", "valid", "correct", "valid_correct"]
REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "analysis/passk_ladder.md"
OUT_JSON = REPO / "analysis/passk_ladder.json"
OUT_FIG = REPO / "reports/figures/passk_ladder"


def main() -> None:
    # Results of models whose dirs are gone (x3 @781/@1562: the SFT script deleted checkpoint dirs with their evals
    # until 2026-10-01) come from the cache, or, if not cached, from the @1/@8/@16/mixed@8 cells of the last table.
    cache = json.loads(OUT_JSON.read_text()) if OUT_JSON.is_file() else {}
    if OUT_MD.is_file():
        for ln in OUT_MD.read_text().splitlines():
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) == len(METRICS) + 1 and cells[0] not in cache and "/" in cells[1]:
                cache[cells[0]] = {m: dict(zip(["@1", "@8", "@16", "mixed@8"], map(float, c.split(" / "))))
                                   for m, c in zip(METRICS, cells[1:])}
    rows = []
    for name, d in MODELS:
        f = d / "rl_gate_dolci_k16/summary.json"
        if f.is_file():
            cache[name] = json.loads(f.read_text())["pass_at_k"]["overall"]
        if name in cache:
            rows.append((name, cache[name]))
    OUT_JSON.write_text(json.dumps(cache, indent=1) + "\n")
    lines = ["# Dolci gate, sampled pass@k (16 samples, T=1.0, 950 items)", "",
             "cells: @1 / @8 / @16 / mixed@8 (fraction of prompts with a non-zero GRPO advantage at 8 rollouts)", "",
             "| model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for name, o in rows:
        lines.append(f"| {name} | " + " | ".join(
            f"{o[m]['@1']:.3f} / {o[m]['@8']:.3f} / {o[m]['@16']:.3f} / {o[m]['mixed@8']:.3f}" for m in METRICS) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, m in zip(axes, ["has_proof", "grammatical", "valid", "valid_correct"]):
        for name, o in rows:
            ks = [k for k in o[m] if k.startswith("@")]
            ax.plot([int(k[1:]) for k in ks], [o[m][k] for k in ks], marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (samples at T=1.0)")
        ax.set_title(f"{m}@k")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("fraction of 950 Dolci gate prompts")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Dolci RL gate: pass@k vs SFT data, lemma catalog and GRPO (2B)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=150)


if __name__ == "__main__":
    main()
