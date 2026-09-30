#!/usr/bin/env python
"""Does the SFT model know which lemmas exist in the RLVL lib? (2026-09-29)

The lib (RLVL-next/rlvl/lib/*.rlvl, DESIGN.md 1.5) is the persistent knowledge base: a proof line
`... ; lib file.name` is checked against the lemma `name:` of lib/file.rlvl. This counts
  lib side    lemmas per lib file, and how many of them any SFT proof cites (pool/train.jsonl, p50 pool)
  model side  every `; lib X` cite in a model's generations, split into
                nonexistent       X is not a lemma of the lib (hallucinated)
                real, unseen      X exists but no SFT proof ever cites it
                real, seen        X exists and SFT proofs cite it
              on the in-domain eval (formal_eval), the tagged benchmarks (formal_bench_tagged) and the
              held-out Dolci gate (rl_gate_dolci), for the 2B/9B p50 SFT models and the GRPO G6 final
              (which stopped citing the lib at all; plotted only when it cites).
Writes analysis/lib_coverage.json and reports/figures/lib_coverage.png.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIB = Path("/vol/home-vol2/ml/laitenbf/RLVL-next/rlvl/lib")
DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
POOL = DATA / "datasets/formal_mixture_20260925/pool/train.jsonl"
SFT = DATA / "formal_mixture_sft_20260925"
GRPO = DATA / "grpo_formal_20260928"
MODELS = [("2B p50 SFT", SFT / "qwen35_2b_dolci_rlvlgen_p50_lr5em6_seed3407"),
          ("9B p50 SFT", SFT / "qwen35_9b_dolci_rlvlgen_p50_lr5em6_seed3407"),
          ("2B G6 frac_hard final", GRPO / "2b_p50_G6_frachard/final"),
          # continued SFT from 2B p50 (scripts/data/build_lemma_continue.py): + lemma catalog vs control
          ("2B p50 + lemma catalog (bf16 DDP)", SFT / "qwen35_2b_p50_cont_lc_lr5em6_seed3407"),
          ("2B p50 + lemma catalog (fp32 master)", SFT / "qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407"),
          ("2B p50 + control", SFT / "qwen35_2b_p50_cont_ct_lr5em6_seed3407"),
          ("2B p50 x3 @781 SFT", SFT / "qwen35_2b_dolci_rlvlgen_p50_x3_lr5em6_seed3407/checkpoint-781"),
          ("2B G7 cvf final (p50 base)", GRPO / "2b_p50_G7_cvf/final"),
          ("2B G8 cvf final (lemma-catalog base)", GRPO / "2b_lcfp32m_G8_cvf/final"),
          ("2B G9 cvf @100 (x3 base)", GRPO / "2b_x3at781_G9_cvf/checkpoint-100")]
EVALS = [("formal_eval", "in-domain"), ("formal_bench_tagged", "tagged benchmarks"), ("rl_gate_dolci", "Dolci gate")]
CITE = re.compile(r"; lib ([A-Za-z_0-9.]+)")


def lib_lemmas() -> dict[str, list[str]]:
    out = {}
    for f in sorted(LIB.glob("*.rlvl")):
        out[f.stem] = [f"{f.stem}.{m.group(1)}" for ln in open(f) if (m := re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*:", ln))]
    return out


def main():
    lib = lib_lemmas()
    names = {n for v in lib.values() for n in v}
    train = collections.Counter()
    for ln in open(POOL):
        train.update(CITE.findall(json.loads(ln)["proof"]))
    seen = {n for n in train if n in names}
    res = {"lib": {f: {"n": len(v), "seen": sum(n in seen for n in v)} for f, v in lib.items()},
           "n_lemmas": len(names), "n_seen": len(seen),
           "train_cites_nonexistent": sum(v for k, v in train.items() if k not in names), "models": {}}
    print(f"lib {len(names)} lemmas, SFT cites {len(seen)}")
    for mname, d in MODELS:
        for ev, _ in EVALS:
            f = d / ev / "generations.jsonl"
            if not f.exists():
                f = d / "final" / ev / "generations.jsonl"   # grpo_gate_ckpts.slurm writes under <run>/final/
            if not f.exists():
                continue
            c = collections.Counter()
            for ln in open(f):
                g = json.loads(ln)
                c.update(CITE.findall(g.get("generation") or g.get("completion") or ""))
            n = sum(c.values())
            r = {"cites": n, "nonexistent": sum(v for k, v in c.items() if k not in names),
                 "real_unseen": sum(v for k, v in c.items() if k in names and k not in seen),
                 "top_nonexistent": [k for k, _ in sorted(((k, v) for k, v in c.items() if k not in names), key=lambda x: -x[1])[:12]]}
            r["real_seen"] = n - r["nonexistent"] - r["real_unseen"]
            res["models"].setdefault(mname, {})[ev] = r
            print(mname, ev, {k: v for k, v in r.items() if k != "top_nonexistent"}, r["top_nonexistent"][:6])
    (REPO / "analysis/lib_coverage.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    fs = list(lib)
    a.bar(fs, [len(lib[f]) for f in fs], color="#ccc", label="lemmas in the lib")
    a.bar(fs, [res["lib"][f]["seen"] for f in fs], color="#2a7", label="cited by >= 1 SFT proof")
    a.set_title(f"lib coverage of the SFT data: {len(seen)}/{len(names)} lemmas ever shown")
    a.tick_params(axis="x", rotation=60)
    a.legend(fontsize=8)
    labels, parts = [], {"real_seen": [], "real_unseen": [], "nonexistent": []}
    for mname, evs in res["models"].items():
        for ev, lbl in EVALS:
            if ev in evs and evs[ev]["cites"]:
                r = evs[ev]
                labels.append(f"{mname}\n{lbl} (n={r['cites']})")
                for k in parts:
                    parts[k].append(r[k] / max(r["cites"], 1))
    bottom = [0.0] * len(labels)
    for k, col, lbl in [("real_seen", "#2a7", "real, seen in SFT"), ("real_unseen", "#fb3", "real, never in SFT"),
                        ("nonexistent", "#d33", "nonexistent (hallucinated)")]:
        b.barh(labels, parts[k], left=bottom, color=col, label=lbl)
        for i, (x0, w) in enumerate(zip(bottom, parts[k])):
            if w > .06:
                b.text(x0 + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7)
        bottom = [x + w for x, w in zip(bottom, parts[k])]
    b.invert_yaxis()
    b.tick_params(axis="y", labelsize=7)
    b.set_xlabel("share of `; lib X` citations")
    b.set_title("what the models cite")
    b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(REPO / "reports/figures/lib_coverage.png", dpi=140)


if __name__ == "__main__":
    main()
