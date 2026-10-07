#!/usr/bin/env python3
"""Rescore GRPO greedy gate generations on the AZ gate_subset_300 with the AZ eval scorer (2026-10-08, Result 16ag).
Run with PYTHONPATH=$S/gen:$S/rlvl_python:scripts, S = checker_snapshot_libext_20261001. Writes analysis/az_subset_vs_grpo.json."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, "/vol/tmp2/laitenbf/synthetic-RLVL/scripts")
from formal_rewards import components
R = Path("/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928")
sub = {r["id"]: r for r in map(json.loads, open("/vol/tmp2/laitenbf/rlvl_data/az/gate_subset_300.jsonl"))}
runs = [("L1_correct", c) for c in ("checkpoint-2500", "checkpoint-4500", "final")] + \
       [("L1_cvf", c) for c in ("checkpoint-2500", "checkpoint-3500", "checkpoint-4500", "checkpoint-4750")] + \
       [("2b_e6_G19_cvffmt_overlong_noproof_kl02", "final")]
out = {}
for arm, c in runs:
    f = R / arm / c / "rl_gate_dolci/generations.jsonl"
    if not f.exists():
        print(arm, c, "missing"); continue
    rows = []
    for g in map(json.loads, open(f)):
        if g["id"] in sub and g.get("sample", 0) == 0:
            cc = components(sub[g["id"]], g["generation"])
            vp = float(cc["valid"]) * float(cc["prem_ok"])
            rows.append((vp, vp * float(cc["correct"]), float(cc["correct"]), g.get("gen_tokens", 0)))
    n = len(rows)
    m = [sum(r[i] for r in rows) / n for i in range(4)]
    out[f"{arm}/{c}"] = {"n": n, "valid_prem": m[0], "cvf": m[1], "correct": m[2], "gen_tokens": m[3]}
    print(f"{arm:40s} {c:16s} n={n} vp={m[0]:.3f} cvf={m[1]:.3f} cor={m[2]:.3f} tok={m[3]:.0f}")
json.dump(out, open("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/az_subset_vs_grpo.json", "w"), indent=1)
