#!/usr/bin/env python3
"""Recompute parse@k from stored samples with the current checker.

The sampler keeps only the first four generations per item, so this rescoring
uses those; it is exact for parse@1 and a lower bound for parse@4. Cheaper
than regenerating, and it makes a checker improvement measurable immediately.
"""
import glob, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from checkers import check_proofwriter

V = os.environ.get("PARSEK_ROOT", os.environ.get("HPCVAULT", "") + "/synthetic-RLVL/lm_eval_results/dose_parsek_20260914")
print("%-34s %4s %8s %8s %8s %8s %9s %9s" % ("model", "d", "pass@1", "maj@4", "parse@1", "parse@4", "valid", "acc|valid"))
for d in sorted(glob.glob(V + "/*/")):
    name = os.path.basename(d.rstrip("/")).replace("qwen25_7b_dose_", "")
    for f in sorted(glob.glob(d + "samples_d*.jsonl")):
        depth = f.split("_d")[-1].split(".")[0]
        rows = [json.loads(l) for l in open(f)]
        if not rows or "texts" not in rows[0]:
            continue
        n = len(rows); p1 = m4 = k1 = k4 = 0; nv = tv = corr_v = 0
        for r in rows:
            gold = r["gold"]; ans = r["answers"][: len(r["texts"])]
            val = [check_proofwriter(r["doc"]["context"], t + "</answer>")["all_valid"] > 0 for t in r["texts"]]
            p1 += ans[0] == gold if ans else 0
            c = Counter(x for x in ans if x)
            m4 += bool(c) and c.most_common(1)[0][0] == gold
            k1 += (val[0] and ans[0] == gold) if val else 0
            first = next((x for v, x in zip(val, ans) if v), "")
            k4 += first == gold
            nv += sum(val); tv += len(val); corr_v += sum(v and x == gold for v, x in zip(val, ans))
        print("%-34s %4s %8.3f %8.3f %8.3f %8.3f %9.3f %9.3f" % (
            name[:34], depth, p1 / n, m4 / n, k1 / n, k4 / n, nv / max(1, tv), corr_v / max(1, nv)))
