from __future__ import annotations

import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1] / "lm_eval_tasks" / "synthrlvl_ood"
sys.path.insert(0, str(TASK_DIR))

from utils import process_longbench_qa_standard, process_longbench_qa_tagged  # noqa: E402


def test_longbench_qa_metrics_use_tagged_answer_and_report_em() -> None:
    doc = {"answers": ["Barack Obama", "Obama"]}
    result = process_longbench_qa_tagged(doc, ["reasoning <answer>Obama</answer>"])

    assert result["qa_f1_score"] == 1.0
    assert result["exact_match"] == 1.0
    assert result["qa_exact_match"] == 1.0
    assert result["tag_found"] == 1.0
    assert result["extracted_nonempty"] == 1.0


def test_longbench_qa_metrics_keep_f1_for_partial_free_form_answers() -> None:
    doc = {"answers": ["Barack Obama"]}
    result = process_longbench_qa_tagged(doc, ["<answer>Obama</answer>"])

    assert 0.0 < result["qa_f1_score"] < 1.0
    assert result["exact_match"] == 0.0


def test_longbench_qa_metrics_do_not_score_unstructured_context_copy() -> None:
    doc = {"answers": ["Paris"]}
    result = process_longbench_qa_tagged(doc, ["The passage says Paris."])

    assert result["qa_f1_score"] == 0.0
    assert result["exact_match"] == 0.0
    assert result["tag_found"] == 0.0


def test_longbench_standard_scores_raw_short_answer() -> None:
    result = process_longbench_qa_standard(
        {"answers": ["Alexander Fleming"]},
        ["Alexander Fleming"],
    )
    assert result == {"score": 1.0, "qa_f1_score": 1.0}
