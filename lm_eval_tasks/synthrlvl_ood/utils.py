from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)(?:\s*</answer>|$)", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")
_LONGBENCH_PREFIX_RE = re.compile(
    r"^\s*Answer the question based on the given passages\.\s*"
    r"Only give me the answer and do not output any other words\.\s*"
    r"The following are given passages\.\s*",
    re.IGNORECASE,
)
_LONGBENCH_SUFFIX_RE = re.compile(
    r"\s*Answer the question based on the given passages\.\s*"
    r"Only give me the answer and do not output any other words\.\s*$",
    re.IGNORECASE,
)
_QUESTION_PREFIX_RE = re.compile(r"^(?:\s*Question:\s*)+", re.IGNORECASE)


def extract_answer(response: Any, *, allow_raw_fallback: bool = True) -> str:
    text = "" if response is None else str(response)
    matches = _ANSWER_RE.findall(text)
    if matches:
        return _clean_extracted(matches[-1])
    for marker in ("Final answer:", "final answer:", "Answer:", "answer:"):
        if marker in text:
            return _clean_extracted(text.rsplit(marker, 1)[-1])
    if not allow_raw_fallback:
        return ""
    return _clean_extracted(text)


def _clean_extracted(text: str) -> str:
    text = re.sub(r"</?(?:think|formal|natural|proof|conclusion|answer)[^>]*>", " ", str(text), flags=re.IGNORECASE)
    return " ".join(text.strip().split())


def _question_block(body: str) -> str:
    return f"<question>\n{str(body).strip()}\n</question>\n"


def _clean_longbench_context(context: Any) -> str:
    text = "" if context is None else str(context)
    text = _LONGBENCH_PREFIX_RE.sub("", text)
    return _LONGBENCH_SUFFIX_RE.sub("", text).strip()


def _clean_longbench_question(question: Any) -> str:
    text = "" if question is None else str(question)
    return _QUESTION_PREFIX_RE.sub("", text).strip()


def doc_to_text_longbench_tagged(doc: dict) -> str:
    return _question_block(
        "Answer the question using the given passages. Put only the final answer in <answer>...</answer>.\n\n"
        f"Passages:\n{_clean_longbench_context(doc.get('context', ''))}\n\n"
        f"Question: {_clean_longbench_question(doc.get('question', ''))}"
    )


def doc_to_text_longbench_standard(doc: dict) -> str:
    passages = _clean_longbench_context(doc.get("context", ""))
    question = _clean_longbench_question(doc.get("question", ""))
    return (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        f"The following are given passages.\n{passages}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        f"Question: {question}\nAnswer:"
    )


def doc_to_text_gsm8k_cot_bare(doc: dict) -> str:
    return _question_block(f"Question: {doc['question']}")


def doc_to_text_gsm8k_cot_prompted(doc: dict) -> str:
    return _question_block(
        f"Question: {doc['question']}\n"
        "Reason through the problem in your learned format, then put the final answer in <answer>...</answer>."
    )


def doc_to_text_longbench_cot_bare(doc: dict) -> str:
    return _question_block(
        f"Passages:\n{_clean_longbench_context(doc.get('context', ''))}\n\n"
        f"Question: {_clean_longbench_question(doc.get('question', ''))}"
    )


def doc_to_text_longbench_cot_prompted(doc: dict) -> str:
    return _question_block(
        f"Passages:\n{_clean_longbench_context(doc.get('context', ''))}\n\n"
        f"Question: {_clean_longbench_question(doc.get('question', ''))}\n"
        "Reason through the problem in your learned format, then put the final answer in <answer>...</answer>."
    )


def _gold_gsm8k_answer(doc: dict) -> str:
    answer = str(doc["answer"])
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    return answer.strip()


def _last_number(text: str) -> str:
    matches = _NUMBER_RE.findall(str(text))
    if not matches:
        return ""
    return _canonical_number(matches[-1])


def _canonical_number(text: str) -> str:
    value = str(text).strip().replace("$", "").replace(",", "")
    if re.fullmatch(r"-?[0-9]+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def process_gsm8k_tagged(doc: dict, results: list[str]) -> dict[str, float]:
    raw = results[0] if results else ""
    extracted = extract_answer(raw, allow_raw_fallback=False)
    tag_found = bool(_ANSWER_RE.search(str(raw)))
    # Score only explicit answer content. Falling back to arbitrary raw numbers
    # in a formal trace can mark <answer>z</answer> correct when the gold number
    # appeared in premises.
    pred = _last_number(extracted) or extracted.strip()
    gold = _canonical_number(_gold_gsm8k_answer(doc))
    return {
        "exact_match": float(pred == gold),
        "tag_found": float(tag_found),
        "extracted_nonempty": float(bool(extracted.strip())),
    }


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / max(1, len(pred_tokens))
    recall = same / max(1, len(gold_tokens))
    return (2 * precision * recall) / (precision + recall)


def qa_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def process_longbench_qa_tagged(doc: dict, results: list[str]) -> dict[str, float]:
    raw = results[0] if results else ""
    extracted = extract_answer(raw, allow_raw_fallback=False)
    best_f1 = 0.0
    best_em = 0.0
    for answer in doc["answers"]:
        best_f1 = max(best_f1, qa_f1_score(extracted, str(answer)))
        best_em = max(best_em, qa_exact_match(extracted, str(answer)))
    return {
        "score": float(best_f1),
        "qa_f1_score": float(best_f1),
        "exact_match": float(best_em),
        "qa_exact_match": float(best_em),
        "tag_found": float(bool(_ANSWER_RE.search(str(raw)))),
        "extracted_nonempty": float(bool(extracted.strip())),
    }


def process_longbench_qa_standard(doc: dict, results: list[str]) -> dict[str, float]:
    """Match lm-eval's stock English LongBench QA-F1 scoring."""
    prediction = str(results[0]).strip() if results else ""
    best_f1 = max(
        (qa_f1_score(prediction, str(answer)) for answer in doc["answers"]),
        default=0.0,
    )
    return {"score": float(best_f1), "qa_f1_score": float(best_f1)}


# --- Graded deduction eval (ProofWriter OWA + deep BranchProof), 2026-08-26 ---

_TFU_RE = re.compile(r"\b(true|false|unknown)\b", re.IGNORECASE)


def doc_to_text_deduction_pw(doc: dict) -> str:
    return (
        f"{str(doc['context']).strip()}\n\n"
        f"Question: {str(doc['question']).strip()}\n"
        "Based only on the statements above, is the claim true, false, or unknown? "
        "Answer with exactly one word: True, False, or Unknown.\n"
        "Answer:"
    )


def process_deduction_pw(doc: dict, results: list[str]) -> dict[str, float]:
    raw = str(results[0]) if results else ""
    match = _TFU_RE.search(raw)
    pred = match.group(1).lower() if match else ""
    gold = str(doc["answer"]).strip().lower()
    return {
        "exact_match": float(bool(pred) and pred == gold),
        "extracted_nonempty": float(bool(pred)),
    }


def doc_to_text_deduction_bp(doc: dict) -> str:
    return (
        f"{str(doc['context']).strip()}\n\n"
        f"Question: {str(doc['question']).strip()}\n"
        "Give only the final answer.\n"
        "Answer:"
    )


def process_deduction_bp(doc: dict, results: list[str]) -> dict[str, float]:
    raw = str(results[0]) if results else ""
    first_line = raw.strip().split("\n", 1)[0]
    pred = normalize_answer(first_line)
    gold = normalize_answer(str(doc["answer"]))
    return {
        "exact_match": float(bool(pred) and pred == gold),
        "extracted_nonempty": float(bool(pred)),
    }


# --- CoT-prompted BP deduction variant, 2026-08-26 ---

_BP_COT_ANSWER_RE = re.compile(r"(?:final answer|answer)\s*:\s*(.+)", re.IGNORECASE)


def doc_to_text_deduction_bp_cot(doc: dict) -> str:
    return (
        f"{str(doc['context']).strip()}\n\n"
        f"Question: {str(doc['question']).strip()}\n"
        "Reason step by step, then give the final answer on its own last line "
        "in the form \"Answer: <answer>\"."
    )


def process_deduction_bp_cot(doc: dict, results: list[str]) -> dict[str, float]:
    raw = str(results[0]) if results else ""
    matches = _BP_COT_ANSWER_RE.findall(raw)
    if matches:
        pred_src = matches[-1].strip().split("\n")[0]
        tag_found = 1.0
    else:
        lines = [l for l in raw.strip().splitlines() if l.strip()]
        pred_src = lines[-1] if lines else ""
        tag_found = 0.0
    pred = normalize_answer(pred_src)
    gold = normalize_answer(str(doc["answer"]))
    return {
        "exact_match": float(bool(pred) and pred == gold),
        "tag_found": tag_found,
        "extracted_nonempty": float(bool(pred)),
    }
