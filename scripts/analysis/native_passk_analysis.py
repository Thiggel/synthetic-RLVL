#!/usr/bin/env python3
"""pass@k, maj@k and parse@k for sampled native-format derivations.

parse@k: take the samples in order, run the checker, and answer with the
first sample whose derivation is entirely valid (every proof line follows,
conclusion equals the last line); no gold is used in the selection. pass@k:
any of the first k samples correct. maj@k: majority answer of the first k.
Also reported: fraction of samples with a fully valid derivation, and the
answer accuracy conditional on validity (does a valid derivation imply the
right answer?).
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from check_native_derivations import check  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lm_eval_tasks", "synthrlvl_ood"))
import utils  # noqa: E402

ANS = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|$)", re.DOTALL)


def answer(text):
    m = ANS.search(text)
    return utils.normalize_answer(m.group(1).strip().split("\n")[0]) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--formal", nargs="*", default=[], help="root substrings whose outputs are formal")
    a = ap.parse_args()
    print("%-34s %4s %7s | %s | %s | %s | %7s %7s" % ("model", "d", "greedy", " ".join("pass@%-2d" % k for k in a.ks),
          " ".join("maj@%-3d" % k for k in a.ks), " ".join("parse@%d" % k for k in a.ks), "valid%", "acc|val"))
    for root in a.roots:
        formal = any(s in root for s in a.formal) or "logic" in os.path.basename(root.rstrip("/"))
        for f in sorted(glob.glob(root + "/native_d*.jsonl"), key=lambda p: int(re.search(r"_d(\d+)", p).group(1))):
            d = int(re.search(r"_d(\d+)", f).group(1))
            rows = [json.loads(l) for l in open(f)]
            g = p = m = pr = 0.0
            g = sum(answer(r["greedy"]) == utils.normalize_answer(r["doc"]["answer"]) for r in rows) / len(rows)
            passk = {k: 0 for k in a.ks}; majk = {k: 0 for k in a.ks}; parsek = {k: 0 for k in a.ks}
            nvalid = ntot = nvalid_correct = 0
            for r in rows:
                gold = utils.normalize_answer(r["doc"]["answer"])
                ans = [answer(s) for s in r["samples"]]
                val = [check(r["doc"], s + "</answer>", formal)["all_valid"] > 0 for s in r["samples"]]
                nvalid += sum(val); ntot += len(val); nvalid_correct += sum(v and x == gold for v, x in zip(val, ans))
                for k in a.ks:
                    passk[k] += any(x == gold for x in ans[:k])
                    c = Counter(x for x in ans[:k] if x)
                    majk[k] += bool(c) and c.most_common(1)[0][0] == gold
                    first = next((x for v, x in zip(val[:k], ans[:k]) if v), "")
                    parsek[k] += first == gold
            n = len(rows)
            print("%-34s %4d %7.3f | %s | %s | %s | %7.3f %7.3f" % (
                os.path.basename(root.rstrip("/"))[:34], d, g,
                " ".join("%7.3f" % (passk[k] / n) for k in a.ks), " ".join("%7.3f" % (majk[k] / n) for k in a.ks),
                " ".join("%7.3f" % (parsek[k] / n) for k in a.ks), nvalid / max(1, ntot), nvalid_correct / max(1, nvalid)))


if __name__ == "__main__":
    main()
