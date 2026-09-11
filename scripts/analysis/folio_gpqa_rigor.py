#!/usr/bin/env python3
"""Paired statistics for FOLIO and GPQA-Diamond against Control.

FOLIO is the out-of-family check on the ProofWriter gain: human-written
first-order-logic problems, 203 items, near-balanced labels. GPQA-Diamond is a
safety check that nothing general was damaged. Both are small, so an
aggregate difference of a few points needs the paired view: which items flip,
McNemar's exact test, and a bootstrap over items. Reads the lm-eval sample
files written on gruenau (default) or alex (--root).
"""
import argparse
import glob
import json
import math
import os
import random
import re
from collections import Counter

ARMS = ["control", "longdoc", "logic_band25", "nl_exact_band25", "condensed_logic_band25",
        "control+sftlogic", "control+sftnl_exact", "logic_band25+sftlogic", "nl_exact_band25+sftnl_exact"]
SHORT = {"control": "Control", "longdoc": "LongDoc", "logic_band25": "Formal",
         "nl_exact_band25": "English", "condensed_logic_band25": "Condensed",
         "control+sftlogic": "Ctl+FSFT", "control+sftnl_exact": "Ctl+ESFT",
         "logic_band25+sftlogic": "Frm+FSFT", "nl_exact_band25+sftnl_exact": "Eng+ESFT"}
FOLIO_LABELS = ["true", "false", "uncertain"]


def find_samples(roots, arm, seed, task):
    suf = "" if seed == 3407 else "_seed%d" % seed
    if "+" in arm:
        if seed != 3407:
            return None
        base, mix = arm.split("+")
        fs = sorted(glob.glob("%s/qwen25_7b_longwin_%s_2p5b_%s_100k_lr5em6/*/samples_%s_*.jsonl" % (roots[0], base, mix, task)))
        return fs[-1] if fs else None
    for root in roots:
        # the 16:43 run under qwen25_longwin_folio_gpqa_20260910 predates the
        # GPQA cap fix (8 generated tokens); its FOLIO samples are fine, its
        # GPQA samples are void
        if task == "synthrlvl_gpqa_diamond" and "qwen25_longwin_folio_gpqa" in root:
            continue
        pat = "%s%s/qwen25_7b_longwin_%s_2p5b_dolci_100k_lr5em6%s/*/samples_%s_*.jsonl" % (root, suf, arm, suf, task)
        fs = sorted(glob.glob(pat))
        if fs:
            return fs[-1]
    return None


def load(path, task):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        gold = str(r["target"]).strip().lower()
        pred = (r["filtered_resps"][0] or "").strip().lower()
        if task == "synthrlvl_folio":
            pred = next((l for l in FOLIO_LABELS if pred.startswith(l)), pred)
        else:  # GPQA: same extraction as the task's process_gpqa
            marked = re.findall(r"(?:answer|final answer)\s*[:\-]?\s*\**\s*\(?([a-d])\)?\b", pred)
            loose = re.findall(r"\b([a-d])\b", pred)
            pred = marked[-1] if marked else (loose[-1] if loose else "")
        neg = bool(re.search(r"\b(not|no|never|neither|nor)\b", str(r["doc"].get("question", "")), re.I))
        rows[r["doc_id"]] = (gold, pred, float(r["exact_match"]), neg)
    return rows


def mcnemar(a, b):
    b_only = c_only = 0
    for k in a:
        if k not in b:
            continue
        if b[k][2] and not a[k][2]:
            b_only += 1
        elif a[k][2] and not b[k][2]:
            c_only += 1
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    k = min(b_only, c_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return b_only, c_only, min(1.0, 2 * p)


def boot_ci(a, b, reps=4000, seed=20260910):
    keys = [k for k in a if k in b]
    diff = [b[k][2] - a[k][2] for k in keys]
    n = len(diff)
    rng = random.Random(seed)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return sum(diff) / n, means[int(0.025 * reps)], means[int(0.975 * reps)]


def balanced(rows, labels):
    rec = []
    for c in labels:
        n = [k for k, v in rows.items() if v[0] == c]
        if n:
            rec.append(sum(rows[k][1] == c for k in n) / len(n))
    return sum(rec) / len(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        os.path.expanduser("~/rlvl_data/lm_eval_results/gruenau_it_folio_gpqa_20260910"),
        os.path.expanduser("~/rlvl_data/lm_eval_results/qwen25_longwin_folio_gpqa_20260910"),
    ])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    args = ap.parse_args()
    for task, labels in (("synthrlvl_folio", FOLIO_LABELS), ("synthrlvl_gpqa_diamond", list("abcd"))):
        for seed in args.seeds:
            data = {}
            for a in ARMS:
                f = find_samples(args.roots, a, seed, task)
                if f:
                    data[a] = load(f, task)
            if "control" not in data:
                continue
            print("=" * 90)
            print("%s  seed %d  (n=%d)" % (task, seed, len(data["control"])))
            print("=" * 90)
            print("%-10s %7s %9s %9s  %s" % ("arm", "acc", "balanced", "n_extr", "pred distribution"))
            for a in ARMS:
                if a not in data:
                    continue
                rows = data[a]
                acc = sum(v[2] for v in rows.values()) / len(rows)
                dist = Counter(v[1] for v in rows.values())
                nonempty = sum(1 for v in rows.values() if v[1])
                print("%-10s %7.3f %9.3f %9d  %s" % (SHORT[a], acc, balanced(rows, labels), nonempty,
                      " ".join("%s=%d" % (k, n) for k, n in sorted(dist.items()))))
            print("\nPaired against Control")
            print("%-10s %8s %8s %12s %26s" % ("arm", "gained", "lost", "McNemar p", "acc diff [95% CI]"))
            for a in ARMS:
                if a == "control" or a not in data:
                    continue
                g, l, p = mcnemar(data["control"], data[a])
                pt, lo, hi = boot_ci(data["control"], data[a])
                print("%-10s %8d %8d %12.3g %11.3f [%.3f, %.3f]" % (SHORT[a], g, l, p, pt, lo, hi))
            if "logic_band25" in data and "nl_exact_band25" in data:
                g, l, p = mcnemar(data["logic_band25"], data["nl_exact_band25"])
                pt, lo, hi = boot_ci(data["logic_band25"], data["nl_exact_band25"])
                print("%-10s %8d %8d %12.3g %11.3f [%.3f, %.3f]   (English vs Formal)" % ("Eng-Form", g, l, p, pt, lo, hi))
            if task == "synthrlvl_folio":
                print("\nQuestion polarity x gold label (the ProofWriter gain lives in negated/false)")
                print("%-26s %5s" % ("cell", "n") + "".join("%10s" % SHORT[a] for a in ARMS if a in data))
                for neg in (True, False):
                    for c in labels:
                        ks = [k for k, v in data["control"].items() if v[0] == c and v[3] == neg]
                        if not ks:
                            continue
                        print("%-26s %5d" % ("%s, gold %s" % ("negated" if neg else "positive", c), len(ks)) + "".join(
                            "%10.3f" % (sum(data[a][k][2] for k in ks) / len(ks)) for a in ARMS if a in data))
                print("\nPer-gold-class accuracy")
                print("%-10s" % "arm" + "".join("%11s" % c for c in labels))
                for a in ARMS:
                    if a not in data:
                        continue
                    rows = data[a]
                    cells = []
                    for c in labels:
                        n = [k for k, v in rows.items() if v[0] == c]
                        cells.append("%6.3f(%3d)" % (sum(rows[k][2] for k in n) / len(n), len(n)) if n else "%11s" % "-")
                    print("%-10s" % SHORT[a] + "".join(cells))
            print()


if __name__ == "__main__":
    main()
