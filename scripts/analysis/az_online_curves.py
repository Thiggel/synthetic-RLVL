#!/usr/bin/env python3
"""Online AlphaZero (scripts/az/az_train.py) learning curves, 2026-10-05.

Reads <run>/metrics.jsonl (one row per self-play iteration) and <run>/evals.jsonl (greedy gate_subset_300 evals) and
draws, per run: search success (z = gold cvf of the committed trajectory, found_any = a gold terminal anywhere in the
tree), value-head ranking quality (online AUC of the root value vs z; held-out AUC at the proof end / middle line),
the losses, and the greedy held-out valid_prem / cvf / correct. Rows written before stats_v 2 hold loss sums over
optimizer steps and are divided by opt_steps here.
Writes analysis/az_online_curves.json and reports/figures/az_online_curves.{png,pdf}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AZ = Path("/vol/tmp2/laitenbf/rlvl_data/az")
RUNS = {"AZ online from e6 (r1, lr 1e-6)": AZ / "online_e6_r1", "AZ online from e6 (r2, lr 3e-6)": AZ / "online_e6_r2_lr3e6",
        "r3: value grad x0.1 into backbone": AZ / "online_e6_r3_vbs01", "r4: value head only (x0)": AZ / "online_e6_r4_vbs0",
        "r5: completed-Q policy target": AZ / "online_e6_r5_cq", "r6: ExIt policy (NTP on found only)": AZ / "online_e6_r6_exit",
        "r7: ExIt, lr 3e-6": AZ / "online_e6_r7_exit_lr3e6",
        "r8: ExIt, 16 sims, 128 expansions": AZ / "online_e6_r8_exit_sims16_exp128",
        "r9: ExIt, lr 3e-6, value grad x0": AZ / "online_e6_r9_exit_lr3e6_vbs0",
        "r10: ExIt, gate-matched pool": AZ / "online_e6_r10_exit_gatepool",
        "r11: Gumbel + subset loss, 2-GPU DP": AZ / "online_e6_r11_gumbel_dp2",
        "r12: Gumbel + subset + ExIt (ntp 4), 2-GPU DP": AZ / "online_e6_r12_gumbel_exit_dp2",
        "r13: Gumbel + subset, positive part only": AZ / "online_e6_r13_gumbel_posonly",
        "a1: r12 recipe, 4-GPU DP on alex, 512 prompts/iter": AZ / "online_e6_a1_gumbel_exit_dp4"}
OUT = REPO / "analysis/az_online_curves.json"
FIG = REPO / "reports/figures/az_online_curves"


def rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = {}
    for r in map(json.loads, open(p)):
        if "iter" in r:
            out[r["iter"]] = r  # a resumed job may repeat an iteration; keep the last
    return [out[k] for k in sorted(out)]


def main():
    res = {}
    for name, d in RUNS.items():
        m = rows(d / "metrics.jsonl")
        for r in m:
            if r.get("stats_v", 1) < 2:
                for k in ("policy_loss", "value_loss"):
                    r[k] = r[k] / max(1, r["opt_steps"])
        res[name] = {"metrics": m, "evals": rows(d / "evals.jsonl")}
    OUT.write_text(json.dumps(res, indent=1) + "\n")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for i, (name, x) in enumerate(res.items()):
        m, e = x["metrics"], x["evals"]
        it = [r["iter"] for r in m]
        c = f"C{i}"
        ax[0, 0].plot(it, [r["found_any"] for r in m], "-", color=c, label=f"{name}: found in tree")
        ax[0, 0].plot(it, [r["z"] for r in m], "--", color=c, label=f"{name}: committed z")
        ax[0, 1].plot(it, [r["value_auc_root_vs_z"] for r in m], "-", color=c, label=f"{name}: online root v vs z")
        ei = [r["iter"] for r in e]
        ax[0, 1].plot(ei, [r["value_auc_end"] for r in e], "o:", color=c, label=f"{name}: held-out, proof end")
        ax[0, 1].plot(ei, [r["value_auc_mid"] for r in e], "s:", color=c, label=f"{name}: held-out, mid proof")
        ax[0, 2].plot(it, [r["value_loss"] for r in m], "-", color=c, label=f"{name}: value BCE")
        ax[0, 2].plot(it, [r["value_abs_err"] for r in m], "--", color=c, label=f"{name}: value |err|")
        ax[1, 0].plot(it, [r["policy_loss"] for r in m], "-", color=c, label=f"{name}: policy CE (+NTP)")
        for k, mk in (("valid_prem", "o-"), ("cvf", "s-"), ("correct", "^-")):
            ax[1, 1].plot(ei, [r[k] for r in e], mk, color=c, label=f"{name}: {k}")
        ax[1, 2].plot(it, [r["iter_s"] / 60 for r in m], "-", color=c, label=f"{name}: iteration (min)")
        ax[1, 2].plot(it, [r["gpu_max_alloc_gb"] for r in m], "--", color=c, label=f"{name}: peak GPU GB")
    titles = ["self-play success (train pool, 256 prompts/iter)", "value head AUC", "value loss",
              "policy loss", "greedy gate_subset_300 (held out)", "cost"]
    for a, t in zip(ax.flat, titles):
        a.set_title(t)
        a.set_xlabel("iteration")
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.suptitle("Online AlphaZero: policy + value head trained jointly from MCTS self-play")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG.with_suffix("." + ext), dpi=130)
    print(json.dumps({k: {"iters": len(v["metrics"]), "evals": len(v["evals"])} for k, v in res.items()}))


if __name__ == "__main__":
    sys.exit(main())
