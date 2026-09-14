#!/usr/bin/env python3
"""Show what the models actually wrote in a parse@k run, and why the checker
accepted or rejected it. Lives in the repo because login-node /tmp is wiped."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from checkers import check_proofwriter

V = os.environ.get("PARSEK_ROOT", os.environ.get("HPCVAULT", "") + "/synthetic-RLVL/lm_eval_results/dose_parsek_20260914")
for d in sorted(glob.glob(V + "/*/")):
    f = os.path.join(d, "samples_d2.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f)]
    r = rows[0]
    print("=" * 80)
    print(os.path.basename(d.rstrip("/")), "| valid flags", r["valid"][:8], "| gold", r["gold"])
    t = r.get("texts", [""])[0]
    print(t[:600])
    res = check_proofwriter(r["doc"]["context"], t + "</answer>")
    print("checker:", {k: round(v, 3) for k, v in res.items()})
    n = sum(1 for x in rows if x.get("texts") and "<proof>" in x["texts"][0])
    print("proof block present in %d of %d first samples" % (n, len(rows)))
