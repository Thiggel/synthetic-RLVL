#!/usr/bin/env python3
"""Do gruenau's evaluations reproduce alex's for the same checkpoints?

Compares every metric that both clusters computed on the seed-3407
instruction-tuned arms. Differences should be within vLLM nondeterminism
(bf16 batching order): a few tenths of a point on generative tasks, zero on
log-likelihood ones. Anything larger means the second cluster is not
comparable and its numbers may not be mixed into the paper's tables.
"""
import glob
import json
import os

D = os.path.expanduser("~/rlvl_data/lm_eval_results")
ARMS = ["control", "longdoc", "logic_band25", "nl_exact_band25", "condensed_logic_band25"]
PAIRS = [  # (alex root, gruenau root)
    ("alex_mirror/qwen25_longwin_downstream_20260907", "gruenau_it_standard_20260910"),
    ("alex_mirror/qwen25_longwin_downstream_20260907", "gruenau_it_multihop_20260910"),
    ("alex_mirror/qwen25_longwin_graded_deduction_20260906", "gruenau_it_deduction_20260910"),
]


def load(root, arm):
    out = {}
    for f in glob.glob("%s/%s/qwen25_7b_longwin_%s_2p5b_dolci_100k_lr5em6*/*/results_*.json" % (D, root, arm)) + \
             glob.glob("%s/%s/qwen25_7b_longwin_%s_2p5b_dolci_100k_lr5em6*/*/*/results_*.json" % (D, root, arm)):
        r = json.load(open(f))["results"]
        for task, m in r.items():
            for k, v in m.items():
                if isinstance(v, float) and "stderr" not in k and k.split(",")[0] in ("exact_match", "acc", "acc_norm", "score", "qa_f1_score"):
                    out[(task, k)] = v
    return out


rows = []
for a_root, g_root in PAIRS:
    for arm in ARMS:
        a, g = load(a_root, arm), load(g_root, arm)
        for key in sorted(set(a) & set(g)):
            rows.append((g_root.split("_")[2], arm, key[0], key[1].split(",")[0], a[key], g[key], g[key] - a[key]))
if not rows:
    print("no overlapping cells yet")
print("%-10s %-24s %-40s %-12s %8s %8s %8s" % ("suite", "arm", "task", "metric", "alex", "gruenau", "diff"))
for r in rows:
    print("%-10s %-24s %-40s %-12s %8.4f %8.4f %+8.4f" % r)
if rows:
    diffs = [abs(r[-1]) for r in rows]
    print("\n%d cells; mean |diff| %.4f, max |diff| %.4f" % (len(rows), sum(diffs) / len(diffs), max(diffs)))
