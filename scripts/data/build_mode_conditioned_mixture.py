#!/usr/bin/env python3
"""Build a mode-conditioned instruction mixture: 90k Dolci + 5k formal + 5k English.

Rationale. Blending derivations into instruction data without a switch was
tried in August: general behaviour was unchanged but the trained scaffold
became brittle, collapsing into repetition loops whenever it was elicited.
The controlled study points at the fix, since mode-conditioned training beat
concatenating both notations by 50.3 against 0.5 percent pass@1. Here the
switch lives in the user turn, so no trainer change is needed: a derivation
example is prefixed with an instruction naming the notation, and Dolci
examples are left exactly as they are. The model therefore answers in prose by
default and emits a checkable derivation only when asked.

Both notations go into one mixture because the point of conditioning is that a
single model can do either, which also halves the number of runs.

Fail-closed: the derivation rows are identified by the signature of their
rendered target, and the count must come out exactly as expected or the build
aborts.
"""
import argparse
import pathlib
import sys

from datasets import Dataset, DatasetDict, load_from_disk

FORMAL_PREFIX = (
    "Answer with a formal derivation: declare the constants and predicates, "
    "list the premises, then give numbered proof lines with their rule "
    "justifications, and finish with the answer field.\n\n"
)
ENGLISH_PREFIX = (
    "Answer with a step-by-step explanation: work through the premises one "
    "inference at a time in plain sentences, and finish with the answer "
    "field.\n\n"
)


def is_derivation(target: str) -> bool:
    """A rendered BranchProof target, in either notation."""
    t = target.strip()
    return ("<answer>" in t) and (t.startswith("<formal>") or t.startswith("<think>"))


def notation_of(target: str) -> str:
    return "formal" if target.strip().startswith("<formal>") else "english"


def split_mixture(ds, expect_derivations):
    keep, deriv = [], []
    for row in ds:
        (deriv if is_derivation(row["target"]) else keep).append(row)
    if len(deriv) != expect_derivations:
        sys.exit("expected %d derivation rows, found %d; refusing to build"
                 % (expect_derivations, len(deriv)))
    return keep, deriv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-mixture", type=pathlib.Path, required=True)
    ap.add_argument("--english-mixture", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--per-notation", type=int, default=5000)
    ap.add_argument("--expect-derivations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()

    formal = load_from_disk(str(a.formal_mixture))
    english = load_from_disk(str(a.english_mixture))

    out = {}
    for split in ("train", "eval"):
        f_split, e_split = formal[split], english[split]
        expect = a.expect_derivations if split == "train" else None

        f_keep, f_deriv = split_mixture(
            f_split, expect if expect is not None else sum(
                1 for r in f_split if is_derivation(r["target"])))
        _, e_deriv = split_mixture(
            e_split, expect if expect is not None else sum(
                1 for r in e_split if is_derivation(r["target"])))

        for row in f_deriv:
            assert notation_of(row["target"]) == "formal", "notation mismatch in formal mixture"
        for row in e_deriv:
            assert notation_of(row["target"]) == "english", "notation mismatch in English mixture"

        n = a.per_notation if split == "train" else min(len(f_deriv), len(e_deriv))
        rows = list(f_keep)  # the Dolci remainder, untouched
        rows += [{"prompt": FORMAL_PREFIX + r["prompt"], "target": r["target"]}
                 for r in f_deriv[:n]]
        rows += [{"prompt": ENGLISH_PREFIX + r["prompt"], "target": r["target"]}
                 for r in e_deriv[:n]]

        ds = Dataset.from_list(rows).shuffle(seed=a.seed)
        out[split] = ds
        print("%-6s %d rows = %d dolci + %d formal + %d english"
              % (split, len(ds), len(f_keep), n, n))

    DatasetDict(out).save_to_disk(str(a.out))
    print("wrote", a.out)

    # the switch must be present on exactly the derivation rows and nowhere else
    tr = out["train"]
    pref = sum(1 for r in tr if r["prompt"].startswith((FORMAL_PREFIX, ENGLISH_PREFIX)))
    deriv = sum(1 for r in tr if is_derivation(r["target"]))
    print("audit: %d prompts carry a mode switch, %d targets are derivations" % (pref, deriv))
    if pref != deriv:
        sys.exit("mode switch and derivation counts disagree")
    print("audit passed")


if __name__ == "__main__":
    main()
