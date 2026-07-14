from __future__ import annotations

import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1] / "lm_eval_tasks" / "synthrlvl_ood"
sys.path.insert(0, str(TASK_DIR))

from utils import (  # noqa: E402
    doc_to_text_longbench_cot_bare,
    doc_to_text_longbench_standard,
    doc_to_text_longbench_tagged,
    process_longbench_qa_standard,
    process_longbench_qa_tagged,
)


def _wrapped_longbench_doc() -> dict:
    return {
        "question": "Question: Where was the scientist born?\n",
        "context": (
            "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
            "The following are given passages.\nPassage 1:\nThe scientist was born in Paris.\n\n"
            "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n"
        ),
        "answers": ["Paris"],
    }


def test_longbench_tagged_prompt_removes_embedded_stock_wrapper() -> None:
    prompt = doc_to_text_longbench_tagged(_wrapped_longbench_doc())

    assert prompt.count("Passage 1:") == 1
    assert "Only give me the answer" not in prompt
    assert "Question: Question:" not in prompt
    assert prompt.count("Question: Where was the scientist born?") == 1


def test_longbench_standard_prompt_reconstructs_stock_wrapper_once() -> None:
    prompt = doc_to_text_longbench_standard(_wrapped_longbench_doc())

    assert prompt.count("The following are given passages.") == 1
    assert prompt.count("Answer the question based on the given passages.") == 2
    assert prompt.count("Question: Where was the scientist born?") == 1
    assert "Question: Question:" not in prompt


def test_longbench_cot_prompt_uses_clean_passages_and_question() -> None:
    prompt = doc_to_text_longbench_cot_bare(_wrapped_longbench_doc())

    assert "Only give me the answer" not in prompt
    assert prompt.count("Passage 1:") == 1
    assert prompt.count("Question: Where was the scientist born?") == 1


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
