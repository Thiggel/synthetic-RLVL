#!/usr/bin/env python3
"""Does search find correct x valid proofs that sampling misses? (Stage 3 round 1, 2026-10-04)

Same e4 student, same round-1 prompts (az/selfplay_r1/prompts_*.jsonl; difficulty = G16@750 teacher cvf rate at
n=16: mixed / zero). Per prompt, "solved" = >= 1 proof with correct * valid_s2 * prem_ok = 1.
  sampling pass@k   rl_prompt_filter.py at n=32, T=1 (cvf reward); pass@k for k < 32 by the unbiased estimator
  MCTS (commit)     mcts_decode --value none --terminal gold (AlphaZero move commitment, 64 expansions x K=8)
  MCTS (solve)      mcts_decode --solve: no commitment, gold-wrong terminals dead, search until a correct proof
Search sampled tokens per prompt are measured; sampling's are 32 x the mean completion length (from the harvest
when the filter does not log it). Writes analysis/az_vs_sampling.{md,json}, reports/figures/az_vs_sampling.{png,pdf}.
"""
from __future__ import annotations

import collections
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
A = Path("/vol/tmp2/laitenbf/rlvl_data/az/selfplay_r1")
SAMPLING = sorted(A.glob("sample_e4_n32*.json"))
SEARCH = {"MCTS (commit)": [A / "mcts_gold_s0", A / "mcts_gold_s1"], "MCTS (solve)": sorted(A.glob("solve_*"))}
OUT, FIG = REPO / "analysis/az_vs_sampling", REPO / "reports/figures/az_vs_sampling"


def pass_at(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def main() -> None:
    recs = {r["id"]: r for p in sorted(A.glob("prompts_*.jsonl")) for r in map(json.loads, open(p))}
    samp: dict[str, dict] = {}
    for f in SAMPLING:
        rates = json.load(open(f))["rates"]
        samp.update({i: v for i, v in rates.items() if i in recs})
    srch: dict[str, dict[str, dict]] = {}
    for name, dirs in SEARCH.items():
        rows = {}
        for d in dirs:
            if (d / "generations.jsonl").is_file():
                rows.update({g["id"]: g for g in map(json.loads, open(d / "generations.jsonl"))})
        if rows:
            srch[name] = rows
    # compare on prompts every available method covers
    ids = [i for i in recs if i in samp and all(i in r for n, r in srch.items() if n != "MCTS (solve)")]
    res = {}
    for name in ["sampling pass@1", "sampling pass@8", "sampling pass@32", *srch]:
        cell = collections.defaultdict(lambda: [0.0, 0])
        for i in ids:
            if name.startswith("sampling"):
                k = int(name.split("@")[1])
                x = pass_at(32, samp[i]["k"], k)
            elif i in srch[name]:
                x = float(srch[name][i]["found"] and srch[name][i]["cvf"] == 1)
            else:
                continue
            for key in ((recs[i]["bench"], recs[i]["difficulty"]), ("all", recs[i]["difficulty"])):
                cell[key][0] += x
                cell[key][1] += 1
        res[name] = {f"{b}|{d}": {"solved": s / n, "n": n} for (b, d), (s, n) in sorted(cell.items())}
        if name in srch:
            g = [srch[name][i] for i in ids if i in srch[name]]
            res[name]["tokens"] = sum(r["gen_tokens"] for r in g) / max(1, len(g))
    # head-to-head on zero prompts: proofs search finds that 32 samples never do
    h2h = {}
    for name, rows in srch.items():
        z = [i for i in ids if i in rows and recs[i]["difficulty"] == "zero"]
        s_only = sum(rows[i]["found"] and rows[i]["cvf"] == 1 and samp[i]["k"] == 0 for i in z)
        p_only = sum(samp[i]["k"] > 0 and not (rows[i]["found"] and rows[i]["cvf"] == 1) for i in z)
        h2h[name] = {"zero_prompts": len(z), "search_only": s_only, "sampling_only": p_only}
    OUT.with_suffix(".json").write_text(json.dumps({"results": res, "head_to_head_zero": h2h, "n_ids": len(ids),
                                                    "sampling_files": [str(f) for f in SAMPLING]}, indent=1) + "\n")
    keys = sorted({k for r in res.values() for k in r if "|" in k})
    lines = ["# Search vs sampling on the round-1 self-play prompts (e4 student)", "",
             f"{len(ids)} prompts covered by sampling (n=32) and MCTS (commit); fraction of prompts with >= 1 correct x "
             "valid_s2 proof. difficulty = G16@750 teacher at n=16 (zero: never solved).", "",
             "| bench / difficulty | n | " + " | ".join(res) + " |", "|---|---:|" + "---:|" * len(res)]
    for k in keys:
        n = next(r[k]["n"] for r in res.values() if k in r)
        lines.append(f"| {k.replace('|', ' / ')} | {n} | " +
                     " | ".join(f"{r[k]['solved']:.3f}" + (f" (n={r[k]['n']})" if r[k]["n"] != n else "")
                                if k in r else "–" for r in res.values()) + " |")
    lines += ["", "Search sampled tokens per prompt: " + ", ".join(f"{n} {r['tokens']:.0f}" for n, r in res.items()
                                                               if "tokens" in r), "",
              "Zero prompts, head to head (prompts solved by one method only): " +
              "; ".join(f"{n}: search-only {h['search_only']}, sampling-only {h['sampling_only']} of {h['zero_prompts']}"
                        for n, h in h2h.items()), ""]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, diff in zip(axes, ("mixed", "zero")):
        ks = [k for k in keys if k.endswith("|" + diff)]
        w = 0.8 / len(res)
        for j, (name, r) in enumerate(res.items()):
            ax.bar([i + j * w for i in range(len(ks))], [r.get(k, {}).get("solved", 0) for k in ks], w, label=name)
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(ks))])
        ax.set_xticklabels([k.split("|")[0] for k in ks], rotation=20)
        ax.set_title(f"{diff} prompts: share with a correct x valid proof")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG.with_suffix("." + ext), dpi=130)


if __name__ == "__main__":
    main()
