#!/usr/bin/env python3
"""Greedy vs test-time MCTS on the 300-item gate subset (az/gate_subset_300.jsonl), per benchmark (2026-10-06).

Greedy: e6 init, r6 @12, r10 @172, r12 @160 = az_train.py evals of online_e6_* (e6 init = r6 iter 0); G19@750 = its rl_gate_dolci
generations rescored on the subset ids with formal_rewards.components (cvf = correct * valid * prem_ok).
MCTS: mcts_decode.py runs in az/mcts_gate300/ (terminal = learned probe or gold cvf; G19 without a value function;
r10 / r12 with their own online value heads, 2026-10-08).
Writes analysis/az_gate_search.{md,json}, reports/figures/az_gate_search.{png,pdf}.
Run with the libext checker snapshot on PYTHONPATH (gen:rlvl_python) plus scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import components  # noqa: E402

D = Path("/vol/tmp2/laitenbf/rlvl_data")
SUB = D / "az/gate_subset_300.jsonl"
M = D / "az/mcts_gate300"
G19 = D / "grpo_formal_20260928/2b_e6_G19_cvffmt_overlong_noproof_kl02/checkpoint-750/rl_gate_dolci/generations.jsonl"
AZ = D / "az"
BENCH = ("dolci_math", "dolci_dapo", "dolci_wordprob", "dolci_yesno", "dolci_knowledge")
MODELS = {"e6 init": ("greedy_e6", "e6_init", "e6_init_gold"),
          "r6 @12 (AZ ExIt)": ("greedy_r6", "r6_ckpt0012", "r6_ckpt0012_gold"),
          "G19 @750 (GRPO)": ("greedy_g19", None, "g19c750_gold_novalue"),
          "r10 @172 (ExIt, gate pool)": ("greedy_r10", "r10_ckpt0172", None),
          "r12 @160 (Gumbel + ExIt)": ("greedy_r12", "r12_ckpt0160", None)}
GREEDY = {"greedy_e6": ("online_e6_r6_exit", 0), "greedy_r6": ("online_e6_r6_exit", 12),
          "greedy_r10": ("online_e6_r10_exit_gatepool", 172), "greedy_r12": ("online_e6_r12_gumbel_exit_dp2", 160)}
OUT, FIG = REPO / "analysis/az_gate_search", REPO / "reports/figures/az_gate_search"


def mcts(name):
    s = json.load(open(M / name / "summary.json"))
    by = {b: {"valid": v["all"]["valid"], "cvf": v["all"]["valid_correct"]} for b, v in s["per_bench"].items()}
    return {"valid": s["decoder"]["valid_s2"], "cvf": s["decoder"]["cvf"], "bench": by}


def greedy_eval(run, it):
    r = next(x for x in map(json.loads, open(AZ / run / "evals.jsonl")) if x.get("iter") == it)
    return {"valid": r["valid_prem"], "cvf": r["cvf"], "bench": {b: {"valid": v["valid_prem"], "cvf": v["cvf"]}
                                                                for b, v in r["bench"].items()}}


def greedy_g19():
    recs = {r["id"]: r for r in map(json.loads, open(SUB))}
    rows = []
    for g in map(json.loads, open(G19)):
        if g["id"] in recs:
            c = components(recs[g["id"]], g["generation"])
            v = float(c["valid"]) * float(c["prem_ok"])
            rows.append((recs[g["id"]]["bench"], v, v * float(c["correct"])))
    agg = lambda rr: {"valid": sum(r[1] for r in rr) / len(rr), "cvf": sum(r[2] for r in rr) / len(rr)}
    return {**agg(rows), "n": len(rows), "bench": {b: agg([r for r in rows if r[0] == b]) for b in BENCH}}


def main():
    res = {k: greedy_eval(*v) for k, v in GREEDY.items()} | {"greedy_g19": greedy_g19()}
    for _, a, b in MODELS.values():
        for n in (a, b):
            if n and (M / n / "summary.json").exists():
                res[n] = mcts(n)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["# Greedy vs test-time MCTS, 300-item gate subset", "",
             "valid = valid with premise numbers stated (greedy) / valid_s2 (MCTS); cvf = correct x that validity.", "",
             "| model | decoding | valid | cvf | " + " | ".join(b.split("_")[1] + " cvf" for b in BENCH) + " |",
             "|---|---|---:|---:|" + "---:|" * len(BENCH)]
    for model, (g, p, o) in MODELS.items():
        for lab, k in (("greedy", g), ("MCTS, probe terminal", p), ("MCTS, gold terminal", o)):
            if k in res:
                x = res[k]
                lines.append(f"| {model} | {lab} | {x['valid']:.3f} | {x['cvf']:.3f} | " +
                             " | ".join(f"{x['bench'].get(b, {}).get('cvf', float('nan')):.3f}" for b in BENCH) + " |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(17, 4.8))
    labs = ("greedy", "MCTS, probe terminal", "MCTS, gold terminal")
    for ax, m in zip(axes, ("valid", "cvf")):
        for i, (model, keys) in enumerate(MODELS.items()):
            for j, k in enumerate(keys):
                if k in res:
                    ax.bar(i + (j - 1) * 0.27, res[k][m], 0.25, color=f"C{j}", label=labs[j])
                    ax.text(i + (j - 1) * 0.27, res[k][m] + 0.005, f"{res[k][m]:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(MODELS)), list(MODELS), fontsize=8)
        ax.set_title({"valid": "validity (premise numbers stated)", "cvf": "cvf = correct x valid"}[m])
        ax.grid(axis="y", alpha=.3)
    h, l_ = axes[1].get_legend_handles_labels()
    uniq = dict(zip(l_, h))
    axes[1].legend(list(uniq.values()), list(uniq), fontsize=8)
    fig.suptitle("Gate subset (n=300): greedy vs MCTS (K=8 lines, 8 sims/move, 64 expansions)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG.with_suffix("." + ext), dpi=130)


if __name__ == "__main__":
    main()
