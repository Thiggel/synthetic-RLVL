#!/usr/bin/env python3
"""Stage 3: search-time decoders vs sampling on the 300-item clean gate subset (2026-10-04).

Model e4 (EI round 4). Subset: rlvl_data/az/gate_subset_300 (60 clean items per Dolci bench). Every variant is rescored
with formal_rewards.components (the cvf reward), so valid_s2 / correct / cvf compare like with like.

  greedy            rl_gate_dolci (one greedy pass)
  k16 mean          rl_gate_dolci_k16, mean over the 16 samples (= the expected single-sample score)
  k16 majority      majority answer over the 16 samples (no checker)
  k16 first-valid   the first of the 16 samples that is Stage-2 valid (best-of-n with the checker as verifier)
  k16 pass@16       oracle: any of the 16 correct (upper bound, not a decoder)
  guided DFS        scripts/guided_decode.py (checker-pruned depth-first line search)
  MCTS <value>/<terminal>  scripts/az/mcts_decode.py (checker-pruned line-level PUCT)
  X + fallback      search proof when found, else the greedy completion

tokens = sampled tokens per item (search: every candidate line). Writes analysis/az_search.{md,json} and
reports/figures/az_search.{png,pdf}.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import components  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
E4 = DATA / "formal_mixture_sft_20260925/qwen35_2b_lc_libext_e4_lr5em6_seed3407/final"
SUB = DATA / "az/gate_subset_300.jsonl"
SEARCH = {"guided DFS": DATA / "az/e4_sub300/guided_dfs", "MCTS none/one": DATA / "az/e4_sub300/mcts_none_one",
          "MCTS probe/probe": DATA / "az/e4_sub300/mcts_probe_probe"}
OUT = REPO / "analysis/az_search"
FIG = REPO / "reports/figures/az_search"


def load(p: Path) -> dict:
    out = collections.defaultdict(list)
    for r in map(json.loads, open(p)):
        out[r["id"]].append(r)
    return out


def main():
    recs = {r["id"]: r for r in map(json.loads, open(SUB))}
    ids = list(recs)
    benches = sorted({r["bench"] for r in recs.values()})

    def score(r):
        c = components(recs[r["id"]], r["generation"])
        return {"valid_s2": float(c["valid"] * c["prem_ok"]), "correct": float(c["correct"]),
                "cvf": float(c["correct"] * c["valid"] * c["prem_ok"]), "tokens": float(r.get("gen_tokens", 0) or 0),
                "found": float(bool(r.get("found")))}

    g_all = load(E4 / "rl_gate_dolci/generations.jsonl")
    k_all = load(E4 / "rl_gate_dolci_k16/generations.jsonl")
    greedy = {i: score(g_all[i][0]) for i in ids}
    per = {"greedy": greedy}
    km, kmaj, kfv, kpass = {}, {}, {}, {}
    for i in ids:
        rs = k_all[i]
        sc = [score(r) for r in rs]
        tok = sum(s["tokens"] for s in sc)
        km[i] = {m: sum(s[m] for s in sc) / len(sc) for m in ("valid_s2", "correct", "cvf", "tokens")}
        ans = collections.Counter(str(r.get("pred_answer")) for r in rs if r.get("pred_answer") is not None)
        j = next((j for j, r in enumerate(rs) if str(r.get("pred_answer")) == ans.most_common(1)[0][0]), 0) if ans else 0
        kmaj[i] = {**sc[j], "tokens": tok}
        j = next((j for j, s in enumerate(sc) if s["valid_s2"]), None)
        kfv[i] = {**(sc[j] if j is not None else sc[0]), "tokens": tok}
        kpass[i] = {"valid_s2": max(s["valid_s2"] for s in sc), "correct": max(s["correct"] for s in sc),
                    "cvf": max(s["cvf"] for s in sc), "tokens": tok}
    per.update({"k16 mean": km, "k16 majority": kmaj, "k16 first-valid": kfv, "k16 pass@16 (oracle)": kpass})
    found_stats = {}
    for name, d in SEARCH.items():
        f = d / "generations.jsonl"
        if not f.exists():
            continue
        s_all = load(f)
        if not all(i in s_all for i in ids):
            continue
        sc = {i: score(s_all[i][0]) for i in ids}
        per[name] = sc
        per[name + " + fallback"] = {i: sc[i] if sc[i]["found"] else {**greedy[i], "tokens": sc[i]["tokens"] + greedy[i]["tokens"]}
                                     for i in ids}
        fi = [i for i in ids if sc[i]["found"]]
        found_stats[name] = {"n_found": len(fi), "correct_found": sum(sc[i]["correct"] for i in fi) / max(1, len(fi)),
                             "greedy_correct_same": sum(greedy[i]["correct"] for i in fi) / max(1, len(fi)),
                             "greedy_correct_notfound": sum(greedy[i]["correct"] for i in ids if i not in fi)
                             / max(1, len(ids) - len(fi))}
    res = {}
    for v, sc in per.items():
        res[v] = {"all": {m: sum(sc[i][m] for i in ids) / len(ids) for m in ("valid_s2", "correct", "cvf", "tokens")}}
        for b in benches:
            bi = [i for i in ids if recs[i]["bench"] == b]
            res[v][b] = {m: sum(sc[i][m] for i in bi) / len(bi) for m in ("valid_s2", "correct", "cvf")}
    OUT.with_suffix(".json").write_text(json.dumps({"variants": res, "found": found_stats}, indent=1) + "\n")
    lines = ["# Stage 3: search decoders vs sampling (e4, 300 clean gate items)", "",
             "valid_s2 = Stage-2 valid (rlvl strict + hardening + stated premise numbers); cvf = valid_s2 · correct; "
             "tokens = sampled tokens per item.", "",
             "| decoder | valid_s2 | correct | cvf | tokens |", "|---|---:|---:|---:|---:|"]
    for v, d in res.items():
        a = d["all"]
        lines.append(f"| {v} | {a['valid_s2']:.3f} | {a['correct']:.3f} | {a['cvf']:.3f} | {a['tokens']:.0f} |")
    lines += ["", "Per bench: valid_s2 / correct / cvf", "", "| decoder | " + " | ".join(b.replace("dolci_", "") for b in benches)
              + " |", "|---|" + "---|" * len(benches)]
    for v, d in res.items():
        lines.append(f"| {v} | " + " | ".join(f"{d[b]['valid_s2']:.2f} / {d[b]['correct']:.2f} / {d[b]['cvf']:.2f}"
                                             for b in benches) + " |")
    lines += [""]
    for name, f in found_stats.items():
        lines.append(f"- **{name}**: found {f['n_found']}/300; correct on found {f['correct_found']:.3f} vs greedy on the "
                     f"same items {f['greedy_correct_same']:.3f} (greedy on not-found items {f['greedy_correct_notfound']:.3f})")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    names = list(res)
    for ax, m in zip(axes[:2], ("cvf", "correct")):
        vals = [res[v]["all"][m] for v in names]
        ax.barh(range(len(names)), vals, color=["C7" if "oracle" in v else "C0" if v.startswith(("greedy", "k16")) else "C3"
                                                for v in names])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if m == "cvf" else [""] * len(names), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{m} (all 300)")
    ax = axes[2]
    for v in names:
        if "oracle" in v:
            continue
        a = res[v]["all"]
        ax.scatter(a["tokens"], a["cvf"], color="C3" if not v.startswith(("greedy", "k16")) else "C0")
        ax.annotate(v, (a["tokens"], a["cvf"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("sampled tokens / item")
    ax.set_ylabel("cvf")
    ax.set_title("cvf vs compute")
    fig.suptitle("Stage 3: checker-pruned search vs sampling on e4 (300 clean Dolci gate items)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG.with_suffix("." + ext), dpi=130)


if __name__ == "__main__":
    main()
