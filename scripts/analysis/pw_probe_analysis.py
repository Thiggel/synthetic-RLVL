#!/usr/bin/env python3
"""Readout of the three ProofWriter probes (gruenau, 2026-09-10).

  mc:   choice log-likelihoods -> threshold-free AUC (gold true vs gold
        false, decidable items), plus accuracy at the argmax. If the arms only
        moved the True/False threshold, AUC is flat and argmax accuracy moves.
  cot:  derive-then-answer prompt -> the polarity x gold 2x2 again, plus how
        often a derivation is written and how long it is.
  pert: flip (claim polarity toggled, gold flips) and ablate (proof premises
        removed, gold unknown), paired with the unperturbed prediction of the
        same item from the graded run.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

D = ("/vol/tmp2/laitenbf/rlvl_data/lm_eval_results")
ARMS = ["control", "longdoc", "logic_band25", "nl_exact_band25", "condensed_logic_band25",
        "control+sftlogic", "control+sftnl_exact", "logic_band25+sftlogic", "nl_exact_band25+sftnl_exact"]
SHORT = {"control": "Control", "longdoc": "LongDoc", "logic_band25": "Formal",
         "nl_exact_band25": "English", "condensed_logic_band25": "Condensed",
         "control+sftlogic": "Ctl+FSFT", "control+sftnl_exact": "Ctl+ESFT",
         "logic_band25+sftlogic": "Frm+FSFT", "nl_exact_band25+sftnl_exact": "Eng+ESFT"}
LABELS = ["true", "false", "unknown"]
DEPTHS = [0, 1, 2, 3, 5]


def run_dir(kind, suite, arm, seed):
    suf = "" if seed == 3407 else "_seed%d" % seed
    if "+" in arm:  # notation SFT run copied from alex (seed 3407 only)
        base, mix = arm.split("+")
        if kind != "it" or seed != 3407:
            return None
        name = "qwen25_7b_longwin_%s_2p5b_%s_100k_lr5em6" % (base, mix)
    else:
        name = "qwen25_7b_longwin_%s_2p5b_%s" % (arm, "base" if kind == "base" else "dolci_100k_lr5em6" + suf)
    d = "%s/gruenau_%s_%s_20260910%s/%s" % (D, kind, suite, suf, name)
    return d if os.path.exists(d + "/.complete") else None


def samples(d, task):
    fs = sorted(glob.glob("%s/*/samples_%s_*.jsonl" % (d, task)))
    return [json.loads(l) for l in open(fs[-1])] if fs else []


def auc(pos, neg):
    """Probability a random gold-true item scores above a random gold-false one."""
    if not pos or not neg:
        return float("nan")
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    rank_sum, i = 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        r = (i + j + 1) / 2.0
        rank_sum += r * sum(1 for k in range(i, j) if s[k][1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def negated(q):
    return " not " in " " + q.lower()


def mc(kind, seed):
    print("\n== MC log-likelihood probe, %s, seed %d ==" % (kind, seed))
    print("%-10s %8s %8s %8s %8s %8s %8s" % ("arm", "acc", "acc_norm", "AUC_t/f", "AUC_neg", "AUC_pos", "pred_true"))
    for arm in ARMS:
        d = run_dir(kind, "deduction_mc", arm, seed)
        if not d:
            continue
        rows = []
        for dep in DEPTHS:
            rows += samples(d, "synthrlvl_deduction_pw_mc_d%d" % dep)
        if not rows:
            continue
        acc = sum(r["acc"] for r in rows) / len(rows)
        accn = sum(r["acc_norm"] for r in rows) / len(rows)
        # margin of True over False, log-likelihood, length-normalised not needed (one token each)
        pos, neg, pos_n, neg_n, pos_p, neg_p, pt = [], [], [], [], [], [], 0
        for r in rows:
            ll = [float(x[0]) for x in r["filtered_resps"]]
            margin = ll[0] - ll[1]
            gold = str(r["target"]) if isinstance(r["target"], str) else r["target"]
            g = LABELS[int(gold)] if str(gold).isdigit() else str(gold).lower()
            pt += int(max(range(3), key=lambda i: ll[i]) == 0)
            if g == "true":
                pos.append(margin); (pos_n if negated(r["doc"]["question"]) else pos_p).append(margin)
            elif g == "false":
                neg.append(margin); (neg_n if negated(r["doc"]["question"]) else neg_p).append(margin)
        print("%-10s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" % (SHORT[arm], acc, accn, auc(pos, neg), auc(pos_n, neg_n), auc(pos_p, neg_p), pt / len(rows)))


def cot(kind, seed):
    print("\n== CoT probe, %s, seed %d ==" % (kind, seed))
    print("%-10s %7s %8s %8s %8s %6s %9s %9s %9s %9s" % ("arm", "acc", "tag", "acc|tag", "words", "loop", "neg/false", "neg/true", "pos/false", "pos/true"))
    for arm in ARMS:
        d = run_dir(kind, "deduction_cot", arm, seed)
        if not d:
            continue
        rows = []
        for dep in DEPTHS:
            rows += samples(d, "synthrlvl_deduction_pw_cot_d%d" % dep)
        if not rows:
            continue
        cell = defaultdict(list)
        for r in rows:
            g = str(r["target"]).lower()
            if g in ("true", "false"):
                cell[("neg" if negated(r["doc"]["question"]) else "pos", g)].append(r["exact_match"])
        f = lambda k: sum(cell[k]) / len(cell[k]) if cell[k] else float("nan")

        def loops(r):  # any non-trivial line repeated three or more times
            c = Counter(l.strip() for l in r["resps"][0][0].splitlines() if len(l.strip()) > 12)
            return 1.0 if c and max(c.values()) >= 3 else 0.0
        tagged = [r for r in rows if r["tag_found"]]
        print("%-10s %7.3f %8.3f %8.3f %8.0f %6.3f %9.3f %9.3f %9.3f %9.3f" % (
            SHORT[arm], sum(r["exact_match"] for r in rows) / len(rows), sum(r["tag_found"] for r in rows) / len(rows),
            sum(r["exact_match"] for r in tagged) / max(1, len(tagged)),
            sum(r["response_words"] for r in rows) / len(rows), sum(loops(r) for r in rows) / len(rows),
            f(("neg", "false")), f(("neg", "true")), f(("pos", "false")), f(("pos", "true"))))


def pert(kind, seed):
    print("\n== Perturbation probes, %s, seed %d ==" % (kind, seed))
    print("%-10s | flip: %6s %9s %9s | ablate: %6s %9s %9s %9s" % ("arm", "acc", "answer", "consist.", "acc", "->false", "->true", "->unk"))
    print("           |       (vs new gold) changed  (same answer to both polarities)")
    for arm in ARMS:
        d = run_dir(kind, "deduction_pert", arm, seed)
        o = run_dir(kind, "deduction", arm, seed)
        if kind == "it" and not o:  # alex ran the same items; its per-item samples are mirrored
            suf = "" if seed == 3407 else "_seed%d" % seed
            cand = "%s/alex_mirror/qwen25_longwin_graded_deduction_20260906%s/qwen25_7b_longwin_%s_2p5b_dolci_100k_lr5em6%s" % (D, suf, arm, suf)
            o = cand if os.path.isdir(cand) else None
        if not d or not o:
            continue
        orig = {}
        for dep in DEPTHS:
            for r in samples(o, "synthrlvl_deduction_pw_d%d" % dep):
                m = re.search(r"\b(true|false|unknown)\b", r["filtered_resps"][0], re.I)
                orig[r["doc"]["source_id"]] = m.group(1).lower() if m else ""
        fl = samples(d, "synthrlvl_deduction_pw_flip")
        ab = samples(d, "synthrlvl_deduction_pw_ablate")

        def pred(r):
            m = re.search(r"\b(true|false|unknown)\b", r["filtered_resps"][0], re.I)
            return m.group(1).lower() if m else ""
        if fl:
            changed = sum(pred(r) != orig.get(r["doc"]["source_id"], "?") for r in fl) / len(fl)
            same = 1 - changed
            facc = sum(r["exact_match"] for r in fl) / len(fl)
        else:
            facc = changed = same = float("nan")
        if ab:
            ab_changed = sum(pred(r) != orig.get(r["doc"]["source_id"], "?") for r in ab) / len(ab)
            sub = [r for r in ab if r["doc"]["orig_answer"] == "false" and negated(r["doc"]["orig_question"])]
            still_false = sum(pred(r) == "false" for r in sub) / max(1, len(sub))
            was_false = sum(orig.get(r["doc"]["source_id"]) == "false" for r in sub) / max(1, len(sub))
            c = Counter(pred(r) for r in ab)
            aacc = sum(r["exact_match"] for r in ab) / len(ab)
            n = len(ab)
            print("%-10s | %13.3f %9.3f %9.3f | %14.3f %9.3f %9.3f %9.3f   changed %.3f; negated/false items: P(false) %.3f before -> %.3f after ablation" % (
                SHORT[arm], facc, changed, same, aacc, c["false"] / n, c["true"] / n, c["unknown"] / n, ab_changed, was_false, still_false))
        else:
            print("%-10s | %13.3f %9.3f %9.3f | (no ablate yet)" % (SHORT[arm], facc, changed, same))
        # the cell that carries the gain: originally negated & gold false -> flipped to positive & gold true
        if fl:
            sub = [r for r in fl if r["doc"]["orig_answer"] == "false" and negated(r["doc"]["orig_question"])]
            if sub:
                print("%-10s   negated/false items after the flip (now positive/true): acc %.3f, n=%d; their original acc %.3f" % (
                    "", sum(r["exact_match"] for r in sub) / len(sub), len(sub),
                    sum(orig.get(r["doc"]["source_id"]) == "false" for r in sub) / len(sub)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["it", "base"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    a = ap.parse_args()
    for kind in a.kinds:
        for seed in a.seeds:
            if kind == "base" and seed != 3407:
                continue
            mc(kind, seed); cot(kind, seed); pert(kind, seed)
