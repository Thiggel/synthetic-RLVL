#!/usr/bin/env python3
"""Materialise GPQA-Diamond as a four-way multiple-choice task and register it.

The canonical repository is gated on the Hub and this machine holds no token,
so this uses the fingertap/GPQA-Diamond mirror, which carries the same 198
items with the four options inline and a letter answer. That is enough for a
control-versus-treatment comparison, which is what we need; absolute numbers
are not comparable to published GPQA scores because the prompt differs.

Fail-closed: every item must expose exactly four labelled options and a gold
letter among A-D, or the build aborts.
"""
import json
import os
import pathlib
import re
import sys
from collections import Counter

VAULT = os.environ.get("HPCVAULT", "/vol/tmp2/laitenbf/rlvl_data")
OUT_DIR = pathlib.Path(VAULT) / "datasets" / "gpqa_diamond_20260910"
OPTION_RE = re.compile(r"(?m)^\s*([A-D])[.)]\s")


def build():
    from datasets import load_dataset

    d = load_dataset("fingertap/GPQA-Diamond")["test"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "gpqa_diamond.jsonl"
    labels = Counter()
    with path.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(d):
            q = str(r["question"]).strip()
            gold = str(r["answer"]).strip().upper()
            opts = sorted(set(OPTION_RE.findall(q)))
            if opts != ["A", "B", "C", "D"]:
                sys.exit("item %d does not expose four options, found %s" % (i, opts))
            if gold not in ("A", "B", "C", "D"):
                sys.exit("item %d has gold %r" % (i, gold))
            labels[gold] += 1
            fh.write(json.dumps({"question": q, "answer": gold, "source_id": i}) + "\n")
    meta = {"n": len(d), "labels": dict(labels), "source": "fingertap/GPQA-Diamond",
            "note": "mirror of GPQA-Diamond; canonical repo is gated", "path": str(path)}
    (OUT_DIR / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote %s: %d items, gold %s" % (path, len(d), dict(labels)))
    return path


UTILS_ADDITION = '''

# --- GPQA-Diamond (four-way multiple choice), 2026-09-10 -------------------

_GPQA_LETTER_RE = re.compile(r"\\b([A-D])\\b")


def doc_to_text_gpqa(doc: dict) -> str:
    return (
        f"{str(doc['question']).strip()}\\n\\n"
        "Answer with the single letter of the correct option.\\n"
        "Answer:"
    )


def process_gpqa(doc: dict, results: list[str]) -> dict[str, float]:
    raw = str(results[0]) if results else ""
    match = _GPQA_LETTER_RE.search(raw.upper())
    pred = match.group(1) if match else ""
    gold = str(doc["answer"]).strip().upper()
    return {
        "exact_match": float(bool(pred) and pred == gold),
        "extracted_nonempty": float(bool(pred)),
    }
'''

YAML = """# GPQA-Diamond, materialised 2026-09-10 from the fingertap mirror because the
# canonical repository is gated. Four-way multiple choice, 198 items, chance
# 25 percent. Used to check that the derivation mixtures do not damage hard
# scientific reasoning; absolute numbers are not comparable to published GPQA.
task: synthrlvl_gpqa_diamond
dataset_path: json
dataset_kwargs:
  data_files:
    test: {jsonl}
test_split: test
output_type: generate_until
doc_to_text: !function utils.doc_to_text_gpqa
doc_to_target: "{{{{answer}}}}"
process_results: !function utils.process_gpqa
generation_kwargs:
  max_gen_toks: 8
  until:
    - "\\n"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 0
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
  - metric: extracted_nonempty
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""


def register(jsonl_path):
    utils = pathlib.Path("lm_eval_tasks/synthrlvl_ood/utils.py")
    s = utils.read_text()
    if "def process_gpqa" in s:
        print("utils already has the GPQA functions")
    else:
        utils.write_text(s + UTILS_ADDITION)
        print("appended GPQA functions to", utils)
    y = pathlib.Path("lm_eval_tasks/synthrlvl_ood/synthrlvl_gpqa_diamond.yaml")
    y.write_text(YAML.format(jsonl=jsonl_path))
    print("wrote", y)


if __name__ == "__main__":
    register(build())
