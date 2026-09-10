#!/usr/bin/env python3
"""Two perturbations of the graded ProofWriter eval items, for causal probes.

The item-level analysis put the whole ProofWriter gain in the negated-claim /
gold-false cell. Two edits of the same items ask what the answer depends on:

  flip:    toggle the polarity of the claim ("The dog is not red." <->
           "The dog is red."), theory unchanged. Gold flips true <-> false.
           A model that reads the negation answers differently on the two
           versions; a negation-blind model gives the same answer to both.
  ablate:  remove from the theory every triple and rule that ProofWriter's
           own proof uses, claim unchanged. The claim becomes underivable, so
           gold is unknown. A model whose "false" depends on the derivation
           should change its answer; one applying a closed-world default
           should not.

Only decidable items (strategies proof / inv-proof) are perturbed. Every
output row keeps the original item's source_id, depth, gold and question so
the stored predictions on the original can be paired with the perturbed ones.
"""
import argparse
import json
import re
from pathlib import Path

DEPTHS = [0, 1, 2, 3, 5]


def negate(q: str) -> str:
    q = q.strip()
    m = re.match(r"^(.*?) is (.+)\.$", q)
    if m and " not " not in q:
        return f"{m.group(1)} is not {m.group(2)}."
    m = re.match(r"^(.*?) (\w+?)s (the .+)\.$", q)
    if m and " does not " not in q:
        return f"{m.group(1)} does not {m.group(2)} {m.group(3)}."
    return ""


def affirm(q: str) -> str:
    q = q.strip()
    m = re.match(r"^(.*?) is not (.+)\.$", q)
    if m:
        return f"{m.group(1)} is {m.group(2)}."
    m = re.match(r"^(.*?) does not (\w+) (the .+)\.$", q)
    if m:
        return f"{m.group(1)} {m.group(2)}s {m.group(3)}."
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="/vol/home-vol2/ml/laitenbf/rlvl_data/datasets/graded_deduction_eval_20260826")
    ap.add_argument("--theory", default="/vol/home-vol2/ml/laitenbf/rlvl_data/datasets/pw_theory.jsonl")
    ap.add_argument("--out", default="/vol/home-vol2/ml/laitenbf/rlvl_data/datasets/proofwriter_perturbations_20260910")
    args = ap.parse_args()
    theory = {}
    for line in open(args.theory):
        r = json.loads(line)
        theory[r["source_id"]] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flips, ablates = [], []
    stats = {"decidable": 0, "flip_fail": 0, "ablate_fail": 0}
    for d in DEPTHS:
        for line in open(f"{args.eval_root}/proofwriter_owa_d{d}.jsonl"):
            item = json.loads(line)
            t = theory[item["source_id"]]
            if t["strategy"] not in ("proof", "inv-proof"):
                continue
            stats["decidable"] += 1
            gold = str(item["answer"]).lower()
            assert item["context"].strip() == t["theory"].strip(), item["source_id"]
            base = dict(source_id=item["source_id"], depth=item["depth"], orig_question=item["question"],
                        orig_answer=gold, strategy=t["strategy"])
            neg = " not " in " " + item["question"]
            newq = affirm(item["question"]) if neg else negate(item["question"])
            if not newq:
                stats["flip_fail"] += 1
            else:
                flips.append(dict(base, context=item["context"], question=newq,
                                  answer="false" if gold == "true" else "true", perturbation="flip"))
            used = [t["triples"].get(u) or t["rules"].get(u) for u in t["used"]]
            used = [u for u in used if u]
            if not used:
                stats["ablate_fail"] += 1
                continue
            sents = [s.strip() for s in re.split(r"(?<=\.)\s+", t["theory"].strip()) if s.strip()]
            kept = [s for s in sents if s not in set(u.strip() for u in used)]
            if len(kept) != len(sents) - len(set(u.strip() for u in used)):
                stats["ablate_fail"] += 1
                continue
            ablates.append(dict(base, context=" ".join(kept), question=item["question"], answer="unknown",
                                perturbation="ablate", removed=len(sents) - len(kept)))
    with open(out / "pw_flip.jsonl", "w") as f:
        for r in flips:
            f.write(json.dumps(r) + "\n")
    with open(out / "pw_ablate.jsonl", "w") as f:
        for r in ablates:
            f.write(json.dumps(r) + "\n")
    stats.update(flip=len(flips), ablate=len(ablates))
    (out / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))
    for r in flips[:3] + ablates[:2]:
        print({k: (v[:160] if isinstance(v, str) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
