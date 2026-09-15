"""Re-score the untagged LongBench QA runs with the stop strings the task
yaml originally lacked.

The standard variants ran with an empty `until` list, so generation continued
past the answer and the F1 of a correct answer was diluted to near zero. The
saved samples hold the raw completion, so the fix does not require a rerun.
"""
import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lm_eval_tasks", "synthrlvl_ood"))
from utils import qa_f1_score  # noqa: E402

STOPS = ("<|im_end|>", "</s>", "\n")


def truncate(text):
    cut = len(text)
    for stop in STOPS:
        i = text.find(stop)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def response_of(sample):
    resp = sample.get("filtered_resps") or sample.get("resps")
    while isinstance(resp, list) and resp:
        resp = resp[0]
    return str(resp)


def score_file(path):
    opener = gzip.open if path.endswith(".gz") else open
    scores = []
    with opener(path, "rt") as handle:
        for line in handle:
            sample = json.loads(line)
            pred = truncate(response_of(sample))
            answers = sample["doc"]["answers"]
            scores.append(max((qa_f1_score(pred, str(a)) for a in answers), default=0.0))
    return sum(scores) / len(scores) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="multihop results directory")
    args = ap.parse_args()
    out = defaultdict(dict)
    for run in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, run)
        if not os.path.isdir(run_dir) or not os.path.exists(os.path.join(run_dir, ".complete")):
            continue
        for dirpath, _, files in os.walk(run_dir):
            for name in files:
                m = re.match(r"samples_(synthrlvl_longbench_\w+_standard)_", name)
                if m:
                    out[run][m.group(1)] = score_file(os.path.join(dirpath, name))
    tasks = sorted({t for r in out.values() for t in r})
    print("run " + " ".join(t.replace("synthrlvl_longbench_", "") for t in tasks))
    for run in sorted(out):
        print(run + " " + " ".join(f"{100 * out[run].get(t, float('nan')):.1f}" for t in tasks))


if __name__ == "__main__":
    main()
