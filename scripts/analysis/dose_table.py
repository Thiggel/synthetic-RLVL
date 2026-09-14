#!/usr/bin/env python3
"""Consolidated dose-sweep table. Reads the evaluation bundles on alex.

Kept in the repository rather than in /tmp, because login-node temporary files
disappear and compute nodes cannot see them anyway.
"""
import collections, glob, json, os

V = os.environ.get("DOSE_ROOT", os.environ.get("HPCVAULT", "/home/vault/c107fa/c107fa12") +
                   "/synthetic-RLVL/lm_eval_results/dose_sweep_20260912")
KEYS = ["deduction_pw_d2", "deduction_pw_d3", "deduction_pw_d5", "folio",
        "deduction_bp_native_d25", "deduction_pw_cot_d3", "deduction_bp_cot_d5",
        "agieval_logiqa_en", "gsm8k", "mmlu", "gpqa_diamond"]
R = collections.defaultdict(dict)
for f in glob.glob(V + "/*/*/*/results_*.json"):
    run = f.split("/dose_sweep_20260912/")[1].split("/")[1]
    for t, m in json.load(open(f))["results"].items():
        if t.startswith(("bbh_", "mmlu_")) and t not in ("bbh", "mmlu"):
            continue
        v = m.get("exact_match,none", m.get("exact_match,strict-match",
            m.get("acc_norm,none", m.get("acc,none"))))
        if v is not None:
            R[run][t.replace("synthrlvl_", "")] = v
runs = sorted(R, key=lambda r: (r.split("_dose_")[1].split("_p")[0], r))
print("%-34s %s" % ("model", " ".join("%-10s" % k.replace("deduction_", "")[:10] for k in KEYS)))
for r in runs:
    print("%-34s %s" % (r.replace("qwen25_7b_dose_", ""),
          " ".join("%-10s" % ("%.3f" % R[r][k] if k in R[r] else "-") for k in KEYS)))
print("\n%d models measured" % len(R))
