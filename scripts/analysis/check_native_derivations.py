#!/usr/bin/env python3
"""Check the derivations the midtrained bases write in the native document format.

BranchProof theories are a Horn fragment: facts "cK is X." and rules
"If cA is X and cA is Y, then cB is Z." (one or two antecedents). A proof line
"cK is X." is valid if it is a premise fact or the head of a rule whose
antecedents are all already established. This checks every line of every
<proof> block in order (stopping at the first invalid line for the prefix
measure), whether the <conclusion> matches the last proof line, whether the
conclusion is actually derivable from the premises by exhaustive forward
chaining, and whether the answer is right. Formal outputs are mapped back to
English through the model's own <predicates> block before checking.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

D = ("/vol/tmp2/laitenbf/rlvl_data/lm_eval_results")
FACT = re.compile(r"^(c\d+) is (\w+)\.?$")
RULE = re.compile(r"^If (c\d+) is (\w+)(?: and (c\d+) is (\w+))?, then (c\d+) is (\w+)\.?$")
FOL_FACT = re.compile(r"^([A-Za-z]\w*)\((c\d+)\)")


def parse_theory(context):
    facts, rules = set(), []
    for line in context.strip().splitlines():
        line = line.strip()
        m = FACT.match(line)
        if m:
            facts.add((m.group(1), m.group(2))); continue
        m = RULE.match(line)
        if m:
            ants = [(m.group(1), m.group(2))]
            if m.group(3):
                ants.append((m.group(3), m.group(4)))
            rules.append((ants, (m.group(5), m.group(6))))
    return facts, rules


def closure(facts, rules):
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for ants, head in rules:
            if head not in known and all(a in known for a in ants):
                known.add(head); changed = True
    return known


def block(text, tag):
    m = re.search(r"<%s>\n?(.*?)\n?</%s>" % (tag, tag), text, re.S)
    return m.group(1) if m else None


def fol_map(text):
    """'Ax: x is ivory' -> {'A': 'ivory'}"""
    pred = block(text, "predicates") or ""
    out = {}
    for line in pred.splitlines():
        m = re.match(r"^\s*([A-Za-z]\w*)\s*(?:x|\(x\))\s*:\s*x is (\w+)", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def proof_lines(text, formal):
    proof = block(text, "proof")
    if proof is None:
        return None
    lines = []
    pm = fol_map(text) if formal else {}
    for raw in proof.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if formal:
            raw = raw.split(";")[0].strip()
            m = FOL_FACT.match(raw)
            if not m or m.group(1) not in pm:
                lines.append(None); continue
            lines.append((m.group(2), pm[m.group(1)]))
        else:
            m = FACT.match(raw)
            lines.append((m.group(1), m.group(2)) if m else None)
    return lines


def check(item, text, formal):
    facts, rules = parse_theory(item["context"])
    derivable = closure(facts, rules)
    lines = proof_lines(text, formal)
    out = dict(has_proof=lines is not None, n_lines=0, all_valid=0.0, valid_prefix=0.0,
               conclusion_derivable=0.0, conclusion_matches_last=0.0, redundant_frac=0.0)
    if lines is None:
        return out
    known = set(facts)
    valid = 0
    first_bad = None
    redundant = 0
    for i, ln in enumerate(lines):
        ok = False
        if ln is not None:
            if ln in known:
                ok = True; redundant += 1  # restating a premise or an earlier line
            else:
                for ants, head in rules:
                    if head == ln and all(a in known for a in ants):
                        ok = True; break
        if ok:
            valid += 1; known.add(ln)
        elif first_bad is None:
            first_bad = i
    n = len(lines)
    out.update(n_lines=n, all_valid=float(first_bad is None and n > 0),
               valid_prefix=(n if first_bad is None else first_bad) / max(1, n),
               redundant_frac=redundant / max(1, n))
    concl = block(text, "conclusion")
    if concl:
        c = concl.strip().splitlines()[-1].strip()
        if formal:
            m = FOL_FACT.match(c.split(";")[0].strip()); pm = fol_map(text)
            cl = (m.group(2), pm.get(m.group(1))) if m else None
        else:
            m = FACT.match(c); cl = (m.group(1), m.group(2)) if m else None
        out["conclusion_derivable"] = float(cl in derivable)
        out["conclusion_matches_last"] = float(bool(lines) and cl == lines[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="base")
    ap.add_argument("--suites", nargs="+", default=["deduction_native", "deduction_native_long"])
    a = ap.parse_args()
    for arm in ["control", "logic_band25", "nl_exact_band25", "longdoc", "condensed_logic_band25",
                "control+sftlogic", "control+sftnl_exact", "logic_band25+sftlogic", "nl_exact_band25+sftnl_exact"]:
        for suite in a.suites:
            if "+" in arm:  # notation SFT runs copied from alex; the mixture decides the notation
                base, mix = arm.split("+")
                formal = mix == "sftlogic"
                if a.kind != "it":
                    continue
                name = "qwen25_7b_longwin_%s_2p5b_%s_100k_lr5em6" % (base, mix)
                formal = mix == "sftlogic"
            else:
                formal = "logic" in arm
                name = "qwen25_7b_longwin_%s_2p5b_%s" % (arm, "base" if a.kind == "base" else "dolci_100k_lr5em6")
            root = "%s/gruenau_%s_%s_20260910/%s" % (D, a.kind, suite, name)
            if not os.path.exists(root + "/.complete"):
                continue
            for f in sorted(glob.glob(root + "/*/samples_synthrlvl_deduction_bp_native*_d*.jsonl")):
                dep = re.search(r"_d(\d+)_", f).group(1)
                agg = defaultdict(float); n = 0
                for line in open(f):
                    r = json.loads(line)
                    res = check(r["doc"], r["resps"][0][0], formal)
                    res["correct"] = r["exact_match"]
                    for k, v in res.items():
                        agg[k] += float(v)
                    n += 1
                print("%-10s %-22s d%-3s n=%d  has_proof %.3f  all_valid %.3f  valid_prefix %.3f  concl_derivable %.3f  concl=last %.3f  redundant %.2f  lines %.0f  correct %.3f" % (
                    a.kind, arm, dep, n, agg["has_proof"] / n, agg["all_valid"] / n, agg["valid_prefix"] / n,
                    agg["conclusion_derivable"] / n, agg["conclusion_matches_last"] / n, agg["redundant_frac"] / n, agg["n_lines"] / n, agg["correct"] / n))


if __name__ == "__main__":
    main()
