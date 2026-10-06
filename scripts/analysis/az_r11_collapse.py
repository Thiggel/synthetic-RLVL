#!/usr/bin/env python3
"""Why r11 (Gumbel + completed-Q + Sampled-MuZero subset loss, no NTP/ExIt term) collapsed (2026-10-07).

Per self-play iteration of az/online_e6_r11_gumbel_dp2 (and r10 = pure ExIt for comparison): share of prompts with a
found proof, mean search-time log-prob of the sampled legal lines (policy entropy proxy), distinct legal lines and
checker-rejected lines per move, and checker fails per expansion by reason. Writes analysis/az_r11_collapse.json and
reports/figures/az_r11_collapse.{png,pdf}.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AZ = Path("/vol/tmp2/laitenbf/rlvl_data/az")
RUNS = {"r11: Gumbel + subset, no anchor": AZ / "online_e6_r11_gumbel_dp2",
        "r10: ExIt only (ntp 4)": AZ / "online_e6_r10_exit_gatepool"}
FIG = REPO / "reports/figures/az_r11_collapse"


def per_iter(d: Path, max_iter: int = 12) -> list[dict]:
    out = []
    for it in range(1, max_iter + 1):
        files = glob.glob(str(d / f"selfplay/iter_{it:04d}*.jsonl"))
        if not files:
            break
        n = found = moves = kids = bad = exp = 0
        lp, fails = [], collections.Counter()
        for f in files:
            for ep in map(json.loads, open(f)):
                n += 1
                found += ep["status"] == "found"
                exp += ep.get("expansions", 0)
                fails.update(ep.get("fail", {}))
                for m in ep.get("moves", []):
                    moves += 1
                    kids += len(m["children"])
                    bad += m.get("n_bad", 0)
                    lp += [c["lp"] for c in m["children"] if c.get("lp") is not None]
        out.append({"iter": it, "found": found / n, "kids_per_move": kids / max(1, moves),
                    "bad_per_move": bad / max(1, moves), "mean_lp": sum(lp) / len(lp) if lp else None,
                    "fail_per_exp": {k: v / max(1, exp) for k, v in fails.items()}})
    return out


def main():
    res = {name: per_iter(d) for name, d in RUNS.items()}
    (REPO / "analysis/az_r11_collapse.json").write_text(json.dumps(res, indent=1) + "\n")
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    for i, (name, rows) in enumerate(res.items()):
        it = [r["iter"] for r in rows]
        ax[0].plot(it, [r["found"] for r in rows], "o-", color=f"C{i}", label=name)
        if any(r["mean_lp"] is not None for r in rows):
            ax[1].plot(it, [r["mean_lp"] for r in rows], "o-", color=f"C{i}", label=name)
        ax[2].plot(it, [r["kids_per_move"] for r in rows], "o-", color=f"C{i}", label=f"{name}: legal lines")
        if any(r["bad_per_move"] for r in rows):  # only subset-loss runs record the rejected lines
            ax[2].plot(it, [r["bad_per_move"] for r in rows], "s--", color=f"C{i}", label=f"{name}: rejected lines")
        for k, ls in (("h_dup_premise", "o-"), ("parse", "s--")):
            ax[3].plot(it, [r["fail_per_exp"].get(k, 0) for r in rows], ls, color=f"C{i}", label=f"{name}: {k}")
    ax[0].set_title("self-play: share of prompts with a valid proof")
    ax[1].set_title("mean log-prob of sampled legal lines\n(lower = flatter policy)")
    ax[2].set_title("distinct lines per move (of K=8 samples)")
    ax[3].set_title("checker fails per expansion")
    for a in ax:
        a.set_xlabel("AZ iteration")
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.suptitle("r11 collapse: without an anchor the subset loss flattens the policy and degenerate lines take over")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG.with_suffix("." + ext), dpi=130)
    for name, rows in res.items():
        print(name)
        for r in rows:
            print(" ", r["iter"], round(r["found"], 3), r["mean_lp"] and round(r["mean_lp"], 2), round(r["kids_per_move"], 2),
                  round(r["bad_per_move"], 2), round(r["fail_per_exp"].get("h_dup_premise", 0), 2))


if __name__ == "__main__":
    main()
