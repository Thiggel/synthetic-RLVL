#!/usr/bin/env python3
"""Where on ProofWriter do the derivation arms gain? Item-level breakdown.

Joins the stored per-item predictions with ProofWriter's own question
metadata (strategy, QDep, QLen, theory family) and asks which kinds of item
flip from wrong to right. A gain that concentrates on items that need a
multi-step proof, or on negated conclusions, is a reasoning gain; a gain that
is flat across strategies and appears on lookup items too would be a
response-style artifact.

ProofWriter strategies: proof / inv-proof (a proof exists; gold true / false),
rconc / inv-rconc (conclusion of a rule that fails to fire; gold Unknown under
OWA), random / inv-random (unrelated fact; gold Unknown). "inv" means the
question is the negation of the derived fact.
"""
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

ARMS = ["control", "longdoc", "logic_band25", "nl_exact_band25", "condensed_logic_band25"]
SHORT = {"control": "Control", "longdoc": "LongDoc", "logic_band25": "Formal",
         "nl_exact_band25": "English", "condensed_logic_band25": "Condensed"}
LABELS = ["true", "false", "unknown"]
DEPTHS = [0, 1, 2, 3, 5]


def load_meta(path):
    meta = {}
    for line in open(path):
        r = json.loads(line)
        meta[r["source_id"]] = r
    return meta


def load_arm(root, arm, suf, meta):
    d = glob.glob("%s/qwen25_7b_longwin_%s_2p5b_dolci_100k_lr5em6%s" % (root, arm, suf))
    if not d:
        return None
    rows = {}
    for dep in DEPTHS:
        fs = sorted(glob.glob(d[0] + "/*/samples_synthrlvl_deduction_pw_d%d_*.jsonl" % dep))
        if not fs:
            continue
        for line in open(fs[-1]):
            r = json.loads(line)
            doc = r["doc"]
            gold = str(r["target"]).strip().lower()
            pred = (r["filtered_resps"][0] or "").strip().lower()
            pred = next((l for l in LABELS if l in pred), "")
            m = meta.get(doc["source_id"], {})
            fam = doc["source_id"].split("-OWA")[0]
            rows[(dep, r["doc_id"])] = dict(
                gold=gold, pred=pred, ok=float(gold == pred), depth=dep,
                strategy=m.get("strategy", "?"), qlen=m.get("QLen", ""),
                family=fam, negated_q=(" not " in " " + doc["question"].lower()),
                n_rules=m.get("n_rules", 0), n_triples=m.get("n_triples", 0),
            )
    return rows


def acc(rows, keys):
    if not keys:
        return float("nan"), 0
    return sum(rows[k]["ok"] for k in keys) / len(keys), len(keys)


def mcnemar(a, b, keys):
    g = sum(1 for k in keys if b[k]["ok"] and not a[k]["ok"])
    l = sum(1 for k in keys if a[k]["ok"] and not b[k]["ok"])
    n = g + l
    if n == 0:
        return g, l, 1.0
    kk = min(g, l)
    p = sum(math.comb(n, i) for i in range(kk + 1)) * 0.5 ** n
    return g, l, min(1.0, 2 * p)


def table(title, data, groups, order=None):
    """groups: name -> list of keys (from control rows)."""
    print("\n" + title)
    names = order or sorted(groups)
    print("%-22s %5s" % ("group", "n") + "".join("%10s" % SHORT[a] for a in ARMS if a in data)
          + "%10s %10s" % ("Eng-Ctl", "p"))
    for g in names:
        keys = groups[g]
        if not keys:
            continue
        cells = "".join("%10.3f" % acc(data[a], keys)[0] for a in ARMS if a in data)
        diff = acc(data["nl_exact_band25"], keys)[0] - acc(data["control"], keys)[0]
        _, _, p = mcnemar(data["control"], data["nl_exact_band25"], keys)
        print("%-22s %5d" % (g, len(keys)) + cells + "%+10.3f %10.2g" % (diff, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser(
        "~/rlvl_data/lm_eval_results/alex_mirror/qwen25_longwin_graded_deduction_20260906"))
    ap.add_argument("--meta", default=("/vol/tmp2/laitenbf/rlvl_data/datasets/pw_qmeta.jsonl"))
    args = ap.parse_args()
    meta = load_meta(args.meta)
    for seed in (3407, 3408):
        suf = "" if seed == 3407 else "_seed3408"
        root = args.root + suf
        data = {}
        for a in ARMS:
            rows = load_arm(root, a, suf, meta)
            if rows:
                data[a] = rows
        if "control" not in data or "nl_exact_band25" not in data:
            continue
        ctl = data["control"]
        keys = [k for k in ctl if all(k in data[a] for a in data)]
        print("=" * 100)
        print("SEED %d   n=%d items, %d with metadata" % (seed, len(keys), sum(ctl[k]["strategy"] != "?" for k in keys)))
        print("=" * 100)

        by = lambda f: defaultdict(list)
        g = defaultdict(list)
        for k in keys:
            g["%s (gold %s)" % (ctl[k]["strategy"], ctl[k]["gold"])].append(k)
        table("Accuracy by ProofWriter strategy", data, g)

        g = defaultdict(list)
        for k in keys:
            if ctl[k]["gold"] in ("true", "false"):
                g["depth %d, gold %s" % (ctl[k]["depth"], ctl[k]["gold"])].append(k)
        table("Decidable items by inference depth and gold label", data, g)

        g = defaultdict(list)
        for k in keys:
            if ctl[k]["gold"] in ("true", "false"):
                g["%s, %s" % (ctl[k]["family"], "negated question" if ctl[k]["negated_q"] else "positive question")].append(k)
        table("Decidable items by theory family and question polarity", data, g)

        g = defaultdict(list)
        for k in keys:
            if ctl[k]["gold"] in ("true", "false") and ctl[k]["qlen"] != "":
                g["proof length %s" % ctl[k]["qlen"]].append(k)
        table("Decidable items by proof length (QLen, number of proof steps)", data, g,
              order=sorted(g, key=lambda s: int(s.split()[-1])))

        g = defaultdict(list)
        for k in keys:
            g["%d rules in theory" % ctl[k]["n_rules"]].append(k)
        table("All items by theory size", data, g, order=sorted(g, key=lambda s: int(s.split()[0])))


        g = defaultdict(list)
        for k in keys:
            if ctl[k]["gold"] in ("true", "false"):
                g["%s question, gold %s" % ("negated" if ctl[k]["negated_q"] else "positive", ctl[k]["gold"])].append(k)
        table("THE 2x2: question polarity x gold label (decidable items). A 'negated -> false' "
              "heuristic gains on negated/false and LOSES on negated/true", data, g)

        g = defaultdict(list)
        for k in keys:
            if ctl[k]["gold"] in ("true", "false") and ctl[k]["negated_q"]:
                g["d%d negated, gold %s" % (ctl[k]["depth"], ctl[k]["gold"])].append(k)
        table("Negated questions by depth and gold", data, g)

        g = defaultdict(list)
        for k in keys:
            if ctl[k]["gold"] in ("true", "false") and not ctl[k]["negated_q"]:
                g["d%d positive, gold %s" % (ctl[k]["depth"], ctl[k]["gold"])].append(k)
        table("Positive questions by depth and gold", data, g)

        print("\nEnglish vs Control flips on decidable items, by strategy x depth")
        print("%-14s %6s %8s %8s %8s" % ("strategy", "depth", "gained", "lost", "net"))
        eng = data["nl_exact_band25"]
        for strat in ("proof", "inv-proof"):
            for dep in DEPTHS:
                ks = [k for k in keys if ctl[k]["strategy"] == strat and ctl[k]["depth"] == dep]
                if not ks:
                    continue
                gg, ll, _ = mcnemar(ctl, eng, ks)
                print("%-14s %6d %8d %8d %+8d" % (strat, dep, gg, ll, gg - ll))
        print("\nPrediction distribution on gold-Unknown items (strategies rconc/random)")
        unk = [k for k in keys if ctl[k]["gold"] == "unknown"]
        for a in ARMS:
            if a not in data:
                continue
            c = defaultdict(int)
            for k in unk:
                c[data[a][k]["pred"] or "(none)"] += 1
            print("%-10s %s" % (SHORT[a], " ".join("%s=%d" % kv for kv in sorted(c.items()))))
        print()


if __name__ == "__main__":
    main()
