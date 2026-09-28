#!/usr/bin/env python
"""Replay the user's fraction reward over the completions real GRPO runs produced (2026-09-28).

For each run's logged completions (<run>/completions/completions_NNNNN.parquet) this recomputes
  frac       (%parsed lines + %checked derived lines + correct) / 3
  frac_hard  frac with the line credit x premise numbers ok x format ok x no circular given
from scripts/formal_rewards.line_stats and the logged `correct`, next to the reward the run was trained on.
The question: does the fraction reward still pay the hacks that lines/gvc runs drifted into?
Writes reports/figures/frac_replay.png. Run with .venv_rlvl_grpo, PYTHONPATH as reward_hack_audit.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import format_ok, line_stats  # noqa: E402

GRPO = Path("/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928")
RUNS = {"G5 lines": ("2b_p50_G5_lines_bal", "lines"), "G3b gvc hardened": ("2b_p50_G3b_gvc_hard", None)}
BLOCK = 10


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(RUNS), figsize=(12, 4.4), sharey=True)
    for ax, (name, (run, trained)) in zip(axes, RUNS.items()):
        pts = []
        files = sorted((GRPO / run / "completions").glob("completions_*.parquet"))
        for b in range(0, 10_000, BLOCK):
            fs = [f for f in files if b < int(f.stem.split("_")[1]) <= b + BLOCK][::2]
            if not fs:
                if b > int(files[-1].stem.split("_")[1]):
                    break
                continue
            acc = {k: 0.0 for k in ["frac", "frac_hard", "trained", "correct", "prem_bad", "fmt"]}
            n = 0
            for f in fs:
                df = pd.read_parquet(f)
                for p, c, cor, tr in zip(df.prompt, df.completion, df.correct,
                                         df[trained] if trained else (df.grammatical + df.valid + df.correct) / 3):
                    q = p.split("<formal>\n", 1)[-1].rsplit("\nassistant", 1)[0]
                    ls, fm = line_stats(q, c), float(format_ok(c))
                    prem = float(ls["n_prem_bad"] == 0)
                    acc["frac"] += (ls["frac_parsed"] + ls["frac_ok"] + cor) / 3
                    acc["frac_hard"] += ((ls["frac_parsed"] + ls["frac_ok"]) * prem * fm * (1 - ls["circular"]) + cor) / 3
                    acc["trained"] += tr
                    acc["correct"] += cor
                    acc["prem_bad"] += 1 - prem
                    acc["fmt"] += fm
                    n += 1
            pts.append((b + BLOCK, {k: v / n for k, v in acc.items()}))
            print(name, b + BLOCK, n, {k: round(v / n, 3) for k, v in acc.items()}, flush=True)
        xs = [x for x, _ in pts]
        for k, lbl, st in [("trained", "trained reward (lines / gvc)", "-"), ("frac", "frac (user)", "-"),
                           ("frac_hard", "frac_hard", "-"), ("correct", "correct", ":"),
                           ("prem_bad", "share with a bad premise number", "--"), ("fmt", "format ok", "--")]:
            ax.plot(xs, [d[k] for _, d in pts], st, label=lbl, marker="o", ms=3)
        ax.set_title(name)
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("mean over logged completions")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("The fraction rewards replayed on real GRPO completions (2B, p50)")
    fig.tight_layout()
    fig.savefig(REPO / "reports/figures/frac_replay.png", dpi=140)


if __name__ == "__main__":
    main()
