#!/usr/bin/env python
"""Training-time validity of Stage-2 GRPO runs, split into generator vs Dolci items (2026-09-29).

--benches gen mixes generator puzzles (formal_mixture pool) into the Dolci training prompts. The SFT
policy proves ~40% of the generator puzzles validly, so these items show whether a reward keeps,
grows or destroys validity where it is reachable. Items are split by matching the prompt text to
the pool. Reads <run>/completions/completions_NNNNN.parquet (the run's own logged components).
Writes analysis/grpo_train_split.json and reports/figures/grpo_train_split.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
POOL = DATA / "datasets/formal_mixture_20260925/pool"
GRPO = DATA / "grpo_formal_20260928"
RUNS = {"G6g frac_hard + gen": "2b_p50_G6g_frachard_gen", "G6h frac_hard hardened + gen": "2b_p50_G6h_frachard_hardened",
        "G6i G6h + restatement fix": "2b_p50_G6i_frachard_restate", "G7 cvf (correct x valid x prem_ok)": "2b_p50_G7_cvf",
        "G7g cvf, gen prompts only": "2b_p50_G7g_cvf_genonly"}
BLOCK = 10


def main():
    pool = {json.loads(ln)["prompt"].strip() for s in ("train", "test") for ln in open(POOL / f"{s}.jsonl")}
    res = {}
    for name, run in RUNS.items():
        fs = sorted((GRPO / run / "completions").glob("completions_*.parquet"))
        for b in range(0, len(fs), BLOCK):
            df = pd.concat(pd.read_parquet(f) for f in fs[b:b + BLOCK])
            REWARD = "frac_hard" if "frac_hard" in df else "cvf"
            q = df.prompt.str.split("<formal>\n", n=1).str[-1].str.rsplit("\nassistant", n=1).str[0].str.strip()
            gen = q.isin(pool)
            r = {"n": len(df), "gen_share": gen.mean(), "reward": df[REWARD].mean()}
            for part, m in (("gen", gen), ("dolci", ~gen)):
                for k in ("valid", "grammatical", "correct"):
                    r[f"{k}_{part}"] = df[k][m].mean()
            res.setdefault(name, []).append((int(fs[b:b + BLOCK][-1].stem.split("_")[1]), {k: float(v) for k, v in r.items()}))
            print(name, res[name][-1][0], {k: round(v, 3) for k, v in r.items()}, flush=True)
    (REPO / "analysis/grpo_train_split.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, part in zip(axes, ("gen", "dolci")):
        for (name, pts), c in zip(res.items(), ("tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange")):
            xs = [x for x, _ in pts]
            ax.plot(xs, [d["reward"] for _, d in pts], "-", c=c, alpha=.45, label=f"{name}: reward (all items)")
            ax.plot(xs, [d[f"valid_{part}"] if d[f"valid_{part}"] == d[f"valid_{part}"] else None for _, d in pts], "-o", ms=3, c=c, label=f"{name}: valid")
            ax.plot(xs, [d[f"correct_{part}"] for _, d in pts], ":", c=c, label=f"{name}: correct")
        ax.set_title({"gen": "generator puzzles (~30% of prompts)", "dolci": "Dolci prompts (~70%)"}[part])
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=.3)
    axes[0].set_ylabel(f"mean over logged completions ({BLOCK}-step blocks)")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, ncol=3, loc="lower center")
    fig.suptitle("Training-time validity: frac_hard (G6*) vs correct x valid x prem_ok (G7*)")
    fig.tight_layout(rect=(0, .19, 1, 1))
    fig.savefig(REPO / "reports/figures/grpo_train_split.png", dpi=140)


if __name__ == "__main__":
    main()
