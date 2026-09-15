#!/usr/bin/env python3
"""Does the negation-repair mechanism hold in the dose sweep?

Splits the stored ProofWriter predictions of every dose model by claim
polarity and gold label. If the mechanism found on the midtrained arms is the
same one operating here, the whole gain sits in the negated / gold-false cell
and the other three cells do not move.
"""
import collections, glob, json, os, re

V = os.environ.get("DOSE_ROOT", os.environ.get("HPCVAULT", "") +
                   "/synthetic-RLVL/lm_eval_results/dose_sweep_20260912")
LAB = ["true", "false", "unknown"]
cells = collections.defaultdict(lambda: collections.defaultdict(list))
pred_true = collections.defaultdict(list)
for f in glob.glob(V + "/deduction/*/*/samples_synthrlvl_deduction_pw_d*.jsonl"):
    run = f.split("/deduction/")[1].split("/")[0]
    for line in open(f):
        r = json.loads(line)
        gold = str(r["target"]).strip().lower()
        pred = (r["filtered_resps"][0] or "").strip().lower()
        pred = next((l for l in LAB if l in pred), "")
        q = r["doc"]["question"]
        neg = " not " in " " + q.lower()
        pred_true[run].append(pred == "true")
        if gold in ("true", "false"):
            cells[run][("negated" if neg else "positive", gold)].append(float(pred == gold))
order = [("negated", "false"), ("negated", "true"), ("positive", "false"), ("positive", "true")]
print("%-30s %s %10s" % ("model", " ".join("%-16s" % ("%s/%s" % c) for c in order), "P(say true)"))
for run in sorted(cells):
    row = []
    for c in order:
        v = cells[run][c]
        row.append("%-16s" % ("%.3f (n=%d)" % (sum(v) / len(v), len(v)) if v else "-"))
    pt = sum(pred_true[run]) / max(1, len(pred_true[run]))
    print("%-30s %s %10.3f" % (run.replace("qwen25_7b_dose_", "")[:30], " ".join(row), pt))
