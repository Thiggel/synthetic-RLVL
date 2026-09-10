#!/usr/bin/env python3
"""Does the ProofWriter gain survive every cheap way of explaining it away?

Four objections a referee will raise, answered from the stored per-item
predictions rather than from aggregate scores:

  1. It is calibration, not reasoning. The control answers "true" on 66 percent
     of items while the gold labels are balanced, so any shift toward balance
     buys accuracy for free. Answered with prior-invariant measures: balanced
     accuracy, macro F1 and Matthews correlation.
  2. It is noise. Answered with McNemar's exact test on paired items and a
     paired bootstrap over items, per depth and pooled.
  3. It is a shallow shortcut. Answered by the depth profile: depth 0 needs
     lookup only and should not move, deeper items should.
  4. It is the intervention's size, not its content. Answered by the
     length-matched LongDoc arm, which replaces the same share with prose
     carrying no deduction.

Run on alex, where the sample files live.
"""
import glob
import json
import math
import os
import random
import sys
from collections import Counter

V = "/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results"
ARMS = ["control", "longdoc", "logic_band25", "nl_exact_band25", "condensed_logic_band25"]
SHORT = {"control": "Control", "longdoc": "LongDoc", "logic_band25": "Formal",
         "nl_exact_band25": "English", "condensed_logic_band25": "Condensed"}
DEPTHS = [0, 1, 2, 3, 5]
LABELS = ["true", "false", "unknown"]


def load(seed):
    """arm -> depth -> {doc_id: (gold, pred)}"""
    root = V + "/qwen25_longwin_graded_deduction_20260906" + ("" if seed == 3407 else "_seed3408")
    suf = "" if seed == 3407 else "_seed3408"
    out = {}
    for a in ARMS:
        d = glob.glob("%s/qwen25_7b_longwin_%s_2p5b_dolci_100k_lr5em6%s" % (root, a, suf))
        if not d:
            continue
        per = {}
        for dep in DEPTHS:
            fs = sorted(glob.glob(d[0] + "/*/samples_synthrlvl_deduction_pw_d%d_*.jsonl" % dep))
            if not fs:
                continue
            rows = {}
            for line in open(fs[-1]):
                r = json.loads(line)
                gold = str(r["target"]).strip().lower()
                pred = (r["filtered_resps"][0] or "").strip().lower()
                pred = next((l for l in LABELS if l in pred), "")
                rows[r["doc_id"]] = (gold, pred)
            per[dep] = rows
        out[a] = per
    return out


def metrics(rows):
    gold = [g for g, _ in rows]
    pred = [p for _, p in rows]
    acc = sum(g == p for g, p in rows) / len(rows)
    # balanced accuracy: mean per-class recall, invariant to the label prior
    recalls = []
    for c in LABELS:
        n = sum(g == c for g in gold)
        if n:
            recalls.append(sum(g == c and p == c for g, p in rows) / n)
    bal = sum(recalls) / len(recalls)
    # macro F1
    f1s = []
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in rows)
        fp = sum(g != c and p == c for g, p in rows)
        fn = sum(g == c and p != c for g, p in rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro_f1 = sum(f1s) / len(f1s)
    # Matthews correlation on the decidable two-class subset
    two = [(g, p) for g, p in rows if g in ("true", "false")]
    tp = sum(g == "true" and p == "true" for g, p in two)
    tn = sum(g == "false" and p == "false" for g, p in two)
    fp = sum(g == "false" and p == "true" for g, p in two)
    fn = sum(g == "true" and p != "true" for g, p in two)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    two_acc = sum(g == p for g, p in two) / len(two) if two else 0.0
    pred_true = sum(p == "true" for p in pred) / len(pred)
    return dict(acc=acc, balanced=bal, macro_f1=macro_f1, mcc=mcc,
                two_class=two_acc, pred_true=pred_true, n=len(rows))


def mcnemar(a_rows, b_rows):
    """Exact two-sided McNemar on paired per-item correctness."""
    b_only = c_only = 0
    for k in a_rows:
        if k not in b_rows:
            continue
        a_ok = a_rows[k][0] == a_rows[k][1]
        b_ok = b_rows[k][0] == b_rows[k][1]
        if b_ok and not a_ok:
            b_only += 1
        elif a_ok and not b_ok:
            c_only += 1
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    k = min(b_only, c_only)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return b_only, c_only, min(1.0, 2 * p)


def boot_ci(a_rows, b_rows, reps=4000, seed=20260910):
    keys = [k for k in a_rows if k in b_rows]
    da = [1.0 if b_rows[k][0] == b_rows[k][1] else 0.0 for k in keys]
    db = [1.0 if a_rows[k][0] == a_rows[k][1] else 0.0 for k in keys]
    diff = [x - y for x, y in zip(da, db)]
    n = len(diff)
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        s = sum(diff[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    point = sum(diff) / n
    return point, means[int(0.025 * reps)], means[int(0.975 * reps)]


def pooled(per, depths):
    out = []
    for d in depths:
        out.extend(per.get(d, {}).values())
    return out


def pooled_keyed(per, depths):
    out = {}
    for d in depths:
        for k, v in per.get(d, {}).items():
            out[(d, k)] = v
    return out


for seed in (3407, 3408):
    data = load(seed)
    if not data:
        continue
    print("=" * 96)
    print("SEED %d" % seed)
    print("=" * 96)
    print("\nPrior-invariant metrics, pooled over inference depths 1-5 (2,000 items)")
    print("%-10s %7s %9s %9s %7s %9s %10s" %
          ("arm", "acc", "balanced", "macro_F1", "MCC", "two-class", "pred_true"))
    infer = [1, 2, 3, 5]
    for a in ARMS:
        if a not in data:
            continue
        m = metrics(pooled(data[a], infer))
        print("%-10s %7.3f %9.3f %9.3f %7.3f %9.3f %10.3f" %
              (SHORT[a], m["acc"], m["balanced"], m["macro_f1"], m["mcc"],
               m["two_class"], m["pred_true"]))

    print("\nPaired tests against Control, pooled over depths 1-5")
    print("%-10s %10s %10s %12s %26s" % ("arm", "gained", "lost", "McNemar p", "acc diff [95% CI]"))
    ctl = pooled_keyed(data["control"], infer)
    for a in ARMS:
        if a == "control" or a not in data:
            continue
        arm = pooled_keyed(data[a], infer)
        g, l, p = mcnemar(ctl, arm)
        pt, lo, hi = boot_ci(ctl, arm)
        print("%-10s %10d %10d %12.2e %11.3f [%.3f, %.3f]" % (SHORT[a], g, l, p, pt, lo, hi))

    print("\nAccuracy by inference depth (depth 0 is lookup, no inference required)")
    print("%-10s" % "arm" + "".join("%9s" % ("d%d" % d) for d in DEPTHS))
    for a in ARMS:
        if a not in data:
            continue
        print("%-10s" % SHORT[a] + "".join(
            "%9.3f" % metrics(list(data[a][d].values()))["acc"] if d in data[a] else "%9s" % "-"
            for d in DEPTHS))

    print("\nBalanced accuracy by depth, the calibration-free view")
    print("%-10s" % "arm" + "".join("%9s" % ("d%d" % d) for d in DEPTHS))
    for a in ARMS:
        if a not in data:
            continue
        print("%-10s" % SHORT[a] + "".join(
            "%9.3f" % metrics(list(data[a][d].values()))["balanced"] if d in data[a] else "%9s" % "-"
            for d in DEPTHS))

    print("\nEnglish vs Control: where the flips happen (pooled depths 1-5)")
    arm = pooled_keyed(data["nl_exact_band25"], infer)
    gained = Counter()
    lost = Counter()
    for k in ctl:
        if k not in arm:
            continue
        c_ok = ctl[k][0] == ctl[k][1]
        a_ok = arm[k][0] == arm[k][1]
        if a_ok and not c_ok:
            gained[ctl[k][0]] += 1
        elif c_ok and not a_ok:
            lost[ctl[k][0]] += 1
    print("  gained by gold label:", dict(gained), " lost by gold label:", dict(lost))
    print()
