#!/usr/bin/env python3
"""Materialise FOLIO as a held-out deduction task and register it with lm-eval.

FOLIO is human-written first-order-logic reasoning: a short set of natural
language premises, a conclusion, and a three-way label. It complements
ProofWriter in two ways that matter here. Its premises are written by people
rather than generated from a template, so it is further from anything the
generator produces; and its labels are close to balanced (72 true, 69
uncertain, 62 false out of 203) where ProofWriter's are not, which makes the
label-bias correction less load-bearing.

The prompt mirrors the ProofWriter task exactly so the two are comparable, and
the answer word is FOLIO's own third label, "uncertain", not ProofWriter's
"unknown".

Materialising to jsonl keeps compute nodes off the network, matching the rest
of the graded-deduction family.
"""
import json
import os
import pathlib
import sys

VAULT = os.environ.get("HPCVAULT", "/home/vault/c107fa/c107fa12")
OUT_DIR = pathlib.Path(VAULT) / "datasets" / "folio_eval_20260910"
REPO = pathlib.Path(__file__).resolve().parents[1] if "__file__" in dir() else pathlib.Path(".")


def build():
    from datasets import load_dataset

    d = load_dataset("tasksource/folio")["validation"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "folio_validation.jsonl"
    n_by_label = {}
    with path.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(d):
            label = str(r["label"]).strip().lower()
            assert label in ("true", "false", "uncertain"), "unexpected label %r" % label
            premises = str(r["premises"]).strip()
            conclusion = str(r["conclusion"]).strip()
            assert premises and conclusion, "empty premises or conclusion at row %d" % i
            n_by_label[label] = n_by_label.get(label, 0) + 1
            fh.write(json.dumps({
                "context": premises,
                "question": conclusion,
                "answer": label,
                "source_id": r.get("example_id", i),
                "story_id": r.get("story_id"),
            }) + "\n")
    meta = {"n": len(d), "labels": n_by_label, "source": "tasksource/folio",
            "split": "validation", "path": str(path)}
    (OUT_DIR / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote %s: %d rows, labels %s" % (path, len(d), n_by_label))
    return path


UTILS_ADDITION = '''

# --- FOLIO (human-written first-order logic), 2026-09-10 -------------------
# Same prompt shape as the ProofWriter task so the two are comparable; FOLIO's
# third label is "uncertain", not ProofWriter's "unknown".

_FOLIO_LABEL_RE = re.compile(r"\\b(true|false|uncertain)\\b", re.IGNORECASE)


def doc_to_text_folio(doc: dict) -> str:
    return (
        f"{str(doc['context']).strip()}\\n\\n"
        f"Question: {str(doc['question']).strip()}\\n"
        "Based only on the statements above, is the claim true, false, or uncertain? "
        "Answer with exactly one word: True, False, or Uncertain.\\n"
        "Answer:"
    )


def process_folio(doc: dict, results: list[str]) -> dict[str, float]:
    raw = str(results[0]) if results else ""
    match = _FOLIO_LABEL_RE.search(raw)
    pred = match.group(1).lower() if match else ""
    gold = str(doc["answer"]).strip().lower()
    return {
        "exact_match": float(bool(pred) and pred == gold),
        "extracted_nonempty": float(bool(pred)),
    }
'''

YAML = """# FOLIO validation split, materialised 2026-09-10 from tasksource/folio.
# Human-written first-order-logic premises with a three-way label, used as a
# held-out external deduction benchmark alongside ProofWriter. Prompt and
# scoring mirror synthrlvl_deduction_pw_*; the third label is "uncertain".
tag:
  - synthrlvl_deduction_graded
task: synthrlvl_folio
dataset_path: json
dataset_kwargs:
  data_files:
    test: {jsonl}
test_split: test
output_type: generate_until
doc_to_text: !function utils.doc_to_text_folio
doc_to_target: "{{{{answer}}}}"
process_results: !function utils.process_folio
generation_kwargs:
  max_gen_toks: 16
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
    if "def process_folio" in s:
        print("utils already has the FOLIO functions")
    else:
        utils.write_text(s + UTILS_ADDITION)
        print("appended FOLIO functions to", utils)
    y = pathlib.Path("lm_eval_tasks/synthrlvl_ood/synthrlvl_folio.yaml")
    y.write_text(YAML.format(jsonl=jsonl_path))
    print("wrote", y)


if __name__ == "__main__":
    p = build()
    register(p)
