#!/usr/bin/env python3
"""Greedy Dolci-gate generations rescored with the hardened GRPO reward, EI SFT arms vs GRPO checkpoints (2026-10-04).

Replaces the ad-hoc analysis/libext_ei_hardened_gate.json (e2/e3/e4 only). Every rl_gate_dolci/generations.jsonl of
the EI SFT arms (e2..e5) and of the GRPO runs G16 (from e2), G17 (from e3), G18 (from e4) is rescored with
formal_rewards.components (new checker): valid = rlvl strict, valid_prem = valid with the premise numbers stated in
their quotes (the cvf reward's validity), cvf = correct * valid_prem, correct = answer matches gold. On all 950 gate
items and on the 713 clean ones (analysis/gate_contamination.md). Missing evals are skipped, so rerun as checkpoints
land. Writes analysis/gate_hardened_rescore.{md,json} and reports/figures/gate_hardened_rescore.{png,pdf}.
Run with the libext checker snapshot on PYTHONPATH (gen:rlvl_python) plus scripts.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import components  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
TEST = DATA / "datasets/rl_gate_dolci_instruct_20260928/test.jsonl"
CONTAM = DATA / "datasets/rl_gate_dolci_instruct_20260928/contamination.json"
SFT = DATA / "formal_mixture_sft_20260925"
GRPO = DATA / "grpo_formal_20260928"
ARMS = {f"{a} (SFT)": SFT / f"qwen35_2b_lc_libext_{a}_lr5em6_seed3407/final" for a in ("e2", "e3", "e4", "e5")}
RUNS = {"G16 (from e2)": GRPO / "2b_e2_G16_cvffmt_overlong_noproof",
        "G17 (from e3)": GRPO / "2b_e3_G17_cvffmt_overlong_noproof",
        "G18 (from e4)": GRPO / "2b_e4_G18_cvffmt_overlong_noproof"}
INIT = {"G16 (from e2)": "e2 (SFT)", "G17 (from e3)": "e3 (SFT)", "G18 (from e4)": "e4 (SFT)"}
METRICS = ("valid", "valid_prem", "cvf", "correct")
OUT = REPO / "analysis/gate_hardened_rescore"
OUT_FIG = REPO / "reports/figures/gate_hardened_rescore"


def rescore(gen: Path, recs: dict, clean: set) -> dict:
    rows = []
    for r in map(json.loads, open(gen)):
        c = components(recs[r["id"]], r["generation"])
        vp = float(c["valid"]) * float(c["prem_ok"])
        rows.append({"id": r["id"], "valid": float(c["valid"]), "valid_prem": vp, "correct": float(c["correct"]),
                     "cvf": vp * float(c["correct"])})
    out = {}
    for sub, rr in (("all", rows), ("clean", [x for x in rows if x["id"] in clean])):
        out[sub] = {"n": len(rr), **{m: round(sum(x[m] for x in rr) / max(1, len(rr)), 4) for m in METRICS}}
    return out


def main():
    recs = {r["id"]: r for r in map(json.loads, open(TEST))}
    clean = set(json.load(open(CONTAM))["clean_ids"])
    res = {}
    for name, d in ARMS.items():
        g = d / "rl_gate_dolci/generations.jsonl"
        if g.exists():
            res[name] = {"step": 0, **rescore(g, recs, clean)}
    for name, d in RUNS.items():
        ck = sorted(((int(m.group(1)), p) for p in d.glob("checkpoint-*")
                     if (m := re.fullmatch(r"checkpoint-(\d+)", p.name))), key=lambda t: t[0])
        for step, p in ck:
            g = p / "rl_gate_dolci/generations.jsonl"
            if g.exists():
                res[f"{name}@{step}"] = {"run": name, "step": step, **rescore(g, recs, clean)}
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")

    lines = ["# Dolci gate (greedy) rescored with the hardened GRPO reward", "",
             "valid = rlvl strict; valid_prem = valid with premise numbers stated (the cvf reward's validity); "
             "cvf = correct · valid_prem. Clean = the 713 gate items without a near-duplicate in the training pool.", "",
             "| model | step | valid | valid_prem | cvf | correct | clean valid_prem | clean cvf | clean correct |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, x in res.items():
        a, c = x["all"], x["clean"]
        lines.append(f"| {name} | {x['step']} | {a['valid']:.3f} | {a['valid_prem']:.3f} | {a['cvf']:.3f} | "
                     f"{a['correct']:.3f} | {c['valid_prem']:.3f} | {c['cvf']:.3f} | {c['correct']:.3f} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for row, sub in zip(axes, ("all", "clean")):
        for ax, m in zip(row, METRICS):
            for i, run in enumerate(RUNS):
                pts = [(0, res[INIT[run]][sub][m])] if INIT[run] in res else []
                pts += [(x["step"], x[sub][m]) for x in res.values() if x.get("run") == run]
                if pts:
                    ax.plot(*zip(*pts), "o-", color=f"C{i}", label=run)
            for j, arm in enumerate(a for a in ARMS if a in res and a not in INIT.values()):
                ax.axhline(res[arm][sub][m], ls="--", color=f"C{3 + j}", label=arm)
            ax.set_title(f"{m} ({sub}, n={next(iter(res.values()))[sub]['n']})")
            ax.set_xlabel("GRPO step (0 = SFT init)")
            ax.grid(alpha=.3)
        row[0].legend(fontsize=7)
    fig.suptitle("Dolci gate, greedy, hardened reward: GRPO from successive EI inits")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG.with_suffix("." + ext), dpi=130)


if __name__ == "__main__":
    main()
