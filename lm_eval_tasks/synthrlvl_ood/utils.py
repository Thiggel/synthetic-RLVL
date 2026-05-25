from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)(?:\s*</answer>|$)", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")


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
    extracted = extract_answer(raw)
    pred = _last_number(extracted) or _last_number(raw) or extracted.strip()
    gold = _canonical_number(_gold_gsm8k_answer(doc))
    return {
        "exact_match": float(pred == gold),
        "tag_found": float(bool(_ANSWER_RE.search(str(raw)))),
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


def process_longbench_qa_tagged(doc: dict, results: list[str]) -> dict[str, float]:
    raw = results[0] if results else ""
    extracted = extract_answer(raw, allow_raw_fallback=False)
    score = 0.0
    for answer in doc["answers"]:
        score = max(score, qa_f1_score(extracted, str(answer)))
    return {
        "score": float(score),
        "qa_f1_score": float(score),
        "tag_found": float(bool(_ANSWER_RE.search(str(raw)))),
        "extracted_nonempty": float(bool(extracted.strip())),
    }
