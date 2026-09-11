#!/usr/bin/env python3
"""Rescore the midtrained bases' standard-prompt multihop answers on the first line.

A base model has no end-of-turn token to emit, so after answering it keeps
writing "Question: ... Answer: ..." continuations. The stock scorer takes the
whole response, which drives token F1 to zero for any base that does not
happen to stop. Scoring the first non-empty line measures the answer itself.
Instruction-tuned models are unaffected (they stop) and are not rescored.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lm_eval_tasks", "synthrlvl_ood"))
import utils  # noqa: E402

D = os.path.expanduser("~/rlvl_data/lm_eval_results/gruenau_base_multihop_20260910")
for arm in ["control", "logic_band25", "nl_exact_band25"]:
    row = []
    for task in ["hotpotqa", "2wikimqa", "musique"]:
        fs = sorted(glob.glob("%s/qwen25_7b_longwin_%s_2p5b_base/*/samples_synthrlvl_longbench_%s_standard_*.jsonl" % (D, arm, task)))
        if not fs:
            row.append("-"); continue
        tot = 0.0; n = 0
        for line in open(fs[-1]):
            r = json.loads(line)
            first = next((l for l in r["resps"][0][0].splitlines() if l.strip()), "")
            tot += max(utils.qa_f1_score(first, str(t)) for t in r["doc"]["answers"]); n += 1
        row.append("%.3f" % (tot / n))
    print("%-16s first-line F1  hotpot %s  2wiki %s  musique %s" % (arm, *row))
