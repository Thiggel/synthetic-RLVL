#!/usr/bin/env python3
"""One table per (kind, suite, seed) from the gruenau evaluation matrix.

Reads every results_*.json under ~/rlvl_data/lm_eval_results/gruenau_* and
prints arm x task tables with the headline metric per task. Base-model
(midtrained only, no chat template) and instruction-tuned numbers are kept in
separate tables and must not be compared to each other.
"""
import glob
import json
import os
import re
from collections import defaultdict

D = os.path.expanduser("~/rlvl_data/lm_eval_results")
ARMS = ["control", "longdoc", "logic_band25", "nl_exact_band25", "condensed_logic_band25"]
SHORT = {"control": "Control", "longdoc": "LongDoc", "logic_band25": "Formal",
         "nl_exact_band25": "English", "condensed_logic_band25": "Condensed"}
HEAD = ["exact_match,none", "exact_match,strict-match", "exact_match,get-answer", "exact_match,flexible-extract", "acc_norm,none", "acc,none", "qa_f1_score,none", "score,none"]


def headline(m):
    for k in HEAD:
        if k in m:
            return k.split(",")[0], m[k]
    return None, None


tables = defaultdict(dict)  # (kind, suite, seed) -> arm -> task -> (metric, value)
for f in glob.glob(D + "/gruenau_*/*/*/results_*.json"):
    m = re.search(r"gruenau_(it|base)_([a-z_]+)_20260910(_seed(\d+))?/qwen25_7b_longwin_([a-z_0-9]+?)_2p5b", f)
    if not m:
        continue
    kind, suite, seed, arm = m.group(1), m.group(2), m.group(4) or "3407", m.group(5)
    done = os.path.exists(os.path.dirname(os.path.dirname(f)) + "/.complete")
    res = json.load(open(f))["results"]
    for task, mm in res.items():
        if task.startswith("bbh_") or task.startswith("mmlu_") or task.startswith("agieval") and task != "agieval_logiqa_en":
            continue
        k, v = headline(mm)
        if k:
            tables[(kind, suite, seed)].setdefault(arm, {})[task] = (k, v, done)

for key in sorted(tables):
    kind, suite, seed = key
    arms = [a for a in ARMS if a in tables[key]]
    tasks = sorted({t for a in arms for t in tables[key][a]})
    print("\n### %s / %s / seed %s%s" % (kind, suite, seed, "" if kind == "it" else "   (midtrained base, no chat template; separate table)"))
    print("| task | metric | " + " | ".join(SHORT[a] for a in arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    for t in tasks:
        cells, metric = [], ""
        for a in arms:
            if t in tables[key][a]:
                k, v, done = tables[key][a][t]
                metric = k
                cells.append("%.3f%s" % (v, "" if done else "*"))
            else:
                cells.append("-")
        print("| %s | %s | %s |" % (t.replace("synthrlvl_", ""), metric, " | ".join(cells)))
print("\n* = bundle not yet marked complete")
