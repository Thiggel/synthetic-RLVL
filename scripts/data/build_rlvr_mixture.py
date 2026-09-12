#!/usr/bin/env python3
"""Build a reasoning-plus-mathematics mixture for verifiable-reward training.

Three kinds, each with a gold answer AND a way to check the written derivation
without the gold:

  kk    Knights and Knaves (K-and-K/knights-and-knaves). Pure logic, no
        arithmetic. The gold is a truth assignment; a derivation is checkable
        by testing the assignment it concludes against the puzzle's own
        statements, which the dataset ships in machine-readable form.
  pw    ProofWriter OWA training items. Deduction over an explicit theory, so
        every proof line can be re-derived by forward chaining.
  math  GSM8K and MATH from the OLMo verifiable-reward mixture, kept as the
        minority so the mixture is reasoning with mathematics alongside, not
        mathematics alone. Checkable only at the level of arithmetic lines.

Prompts use the midtraining document format, which is what makes these models
write a derivation at all.
"""
import argparse, json, random, re
from pathlib import Path


def wrap(lines, question):
    numbered = "\n".join(f"{i + 1}. {l.strip()}" for i, l in enumerate(lines) if l.strip())
    return f"<question>\n{numbered}\n{question.strip()}\n</question>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vol/tmp2/laitenbf/rlvl_data/datasets/rlvr_mixture_20260912")
    ap.add_argument("--per-kind", type=int, default=5000)
    ap.add_argument("--pw-root", default="/vol/tmp2/laitenbf/rlvl_data/datasets/proofwriter_raw")
    ap.add_argument("--seed", type=int, default=20260912)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from datasets import load_dataset
    rows = []

    # --- knights and knaves ------------------------------------------------
    for split in ["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"]:
        for r in load_dataset("K-and-K/knights-and-knaves", "train", split=split):
            quiz = r["quiz"]
            sents = re.split(r"(?<=[.?!])\s+", quiz.strip())
            q = "Who is a knight and who is a knave?"
            rows.append(dict(kind="kk", prompt=wrap(sents, q),
                             answer=r["solution_text"], names=r["names"],
                             solution=r["solution"], statements=str(r["statements"]),
                             source=f"kk_{split}"))
    # --- proofwriter -------------------------------------------------------
    pw = Path(a.pw_root)
    for d in (1, 2, 3, 5):
        f = pw / f"depth-{d}" / "meta-train.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            theory = r["theory"].strip()
            sents = re.split(r"(?<=\.)\s+", theory)
            for qk, q in r["questions"].items():
                ans = str(q["answer"]).lower()
                if ans not in ("true", "false", "unknown"):
                    continue
                rows.append(dict(kind="pw", prompt=wrap(sents, "Is the following claim true, false, or unknown: " + q["question"].strip()),
                                 answer=ans, context=theory, question=q["question"],
                                 source=f"pw_d{d}"))
    # --- mathematics -------------------------------------------------------
    mix = load_dataset("allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", split="train")
    for r in mix:
        if r["dataset"] not in ("gsm8k", "MATH"):
            continue
        text = r["messages"][0]["content"]
        text = re.sub(r"^Question:\s*", "", text.strip())
        sents = re.split(r"(?<=[.?!])\s+", text)
        body, q = (sents[:-1], sents[-1]) if len(sents) > 1 else (sents, "What is the answer?")
        rows.append(dict(kind="math", prompt=wrap(body, q), answer=str(r["ground_truth"]).strip(),
                         source=r["dataset"]))

    out = {}
    for k in ("kk", "pw", "math"):
        sel = [r for r in rows if r["kind"] == k]
        rng.shuffle(sel)
        out[k] = sel[: a.per_kind]
        print(f"{k}: {len(sel)} available, {len(out[k])} kept")
    final = out["kk"] + out["pw"] + out["math"]
    rng.shuffle(final)
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.jsonl", "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    (d / "manifest.json").write_text(json.dumps(
        dict(seed=a.seed, per_kind=a.per_kind, total=len(final),
             counts={k: len(v) for k, v in out.items()}), indent=2))
    print("wrote", d / "train.jsonl", len(final), "rows")


if __name__ == "__main__":
    main()
