#!/usr/bin/env python3
"""Rescore MATH-500 from the answer prefix with symbolic equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


SCHEMA_VERSION = 4
SIDECAR_NAME = "math500_answer_prefix_math_verify.json"
METRIC_NAME = "answer_prefix_math_verify,none"

_EXTRACTION_CONFIG = (LatexExtractionConfig(), ExprExtractionConfig())
_EQUALITY_RE = re.compile(r"(?<![<>!])=(?!=)")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?"
_NUMBER_RE = re.compile(rf"(?<![A-Za-z\\]){_NUMBER}")
_LEADING_NUMBER_RE = re.compile(rf"^{_NUMBER}")
_NUMBER_LIST_RE = re.compile(rf"^({_NUMBER}(?:\s*,\s*{_NUMBER})+)\s*\.?\s*$")
_TRAILING_OPERATOR_RE = re.compile(r"(?:[+\-*/=]|\\(?:cdot|div|times))\s*$")
_NEXT_PROMPT_RE = re.compile(r"^(?:problem|question)\s*:", re.IGNORECASE)
_ANSWER_CUE_RE = re.compile(r"\b(?:answer|solution|therefore|thus|hence)\b", re.IGNORECASE)
_MATH_SPAN_RE = re.compile(r"(?<!\\)\$((?:\\.|[^$])*)(?<!\\)\$")
_UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")
_TRAILING_PERCENT_RE = re.compile(r"(?:\\%|%)\s*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_response(row: dict[str, Any]) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered and isinstance(filtered[0], str):
        return filtered[0]
    responses = row.get("resps")
    if (
        isinstance(responses, list)
        and responses
        and isinstance(responses[0], list)
        and responses[0]
        and isinstance(responses[0][0], str)
    ):
        return responses[0][0]
    return ""


def first_answer_line(response: str) -> str:
    return next((line.strip() for line in response.splitlines() if line.strip()), "")


def _strip_terminal_period(value: str) -> str:
    value = value.strip()
    return value[:-1].rstrip() if value.endswith(".") else value


def _complete_leading_math(line: str) -> tuple[str | None, bool]:
    if not line.startswith("$"):
        return None, False
    match = _MATH_SPAN_RE.match(line)
    if match is None:
        return None, False
    return match.group(1).strip(), True


def _last_explicit_token(line: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    math_spans: list[tuple[int, int]] = []
    for match in _MATH_SPAN_RE.finditer(line):
        candidates.append((match.end(), match.group(1).strip()))
        math_spans.append((match.start(), match.end()))
    for match in _NUMBER_RE.finditer(line):
        if any(start <= match.start() < end for start, end in math_spans):
            continue
        candidates.append((match.end(), match.group(0).strip()))
    return max(candidates, default=(0, None))[1]


def _final_explicit_answer(target: str, response: str) -> str | None:
    target_is_equation = bool(_EQUALITY_RE.search(target))
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines[1:], start=1):
        if _NEXT_PROMPT_RE.match(line) or len(_UNESCAPED_DOLLAR_RE.findall(line)) % 2:
            continue
        candidate = _last_explicit_token(line)
        if not candidate:
            continue
        candidate = _strip_terminal_period(candidate)
        if not target_is_equation and _EQUALITY_RE.search(candidate):
            candidate = _strip_terminal_period(_EQUALITY_RE.split(candidate)[-1])
        if candidate and not _TRAILING_OPERATOR_RE.search(candidate):
            has_cue = bool(_ANSWER_CUE_RE.search(line))
            has_math = bool(_MATH_SPAN_RE.search(line))
            priority = 2 if has_cue and has_math else 1 if has_cue else 0
            candidates.append((priority, index, candidate))
    return max(candidates)[2] if candidates else None


def extract_answer_prefix(target: str, response: str) -> tuple[str | None, str]:
    """Extract the answer requested immediately after the benchmark's ``Answer:`` prompt."""

    line = first_answer_line(response)
    if not line:
        return None, "empty_response"
    if len(_UNESCAPED_DOLLAR_RE.findall(line)) % 2:
        return None, "unbalanced_math_delimiter"

    math_prefix, wrapped = _complete_leading_math(line)
    candidate = math_prefix if wrapped else line
    candidate = _strip_terminal_period(candidate or "")
    if not candidate or _TRAILING_OPERATOR_RE.search(candidate):
        return None, "incomplete_answer_line"

    target_is_equation = bool(_EQUALITY_RE.search(target))
    if target_is_equation:
        return candidate, "full_equation"

    if _EQUALITY_RE.search(candidate):
        rhs = _strip_terminal_period(_EQUALITY_RE.split(candidate)[-1])
        if not rhs or _TRAILING_OPERATOR_RE.search(rhs):
            return None, "incomplete_equality"
        if not re.search(r"[A-Za-z]{3,}", rhs):
            return rhs, "equality_rhs"

    if line.startswith(("(", "[", r"\left(")):
        return candidate, "leading_tuple_or_interval"
    if wrapped:
        return candidate, "leading_math"

    number_list = _NUMBER_LIST_RE.match(line)
    if number_list:
        return number_list.group(1), "leading_number_list"

    leading_number = _LEADING_NUMBER_RE.match(line)
    if leading_number:
        return _strip_terminal_period(leading_number.group(0)), "leading_number"

    if re.fullmatch(r"\(?[A-Fa-f]\)?", line.strip()):
        return line.strip(), "multiple_choice_letter"

    token = _last_explicit_token(line)
    if token:
        return token, "last_explicit_token"
    final_answer = _final_explicit_answer(target, response)
    if final_answer:
        return final_answer, "final_explicit_token"
    return None, "no_answer_token"


@lru_cache(maxsize=4096)
def _parse_expression(value: str) -> list[Any]:
    return parse(
        f"${value}$",
        extraction_config=_EXTRACTION_CONFIG,
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )


def equivalent(target: str, candidate: str | None) -> tuple[bool, str | None]:
    if candidate is None:
        return False, None
    try:
        # Match the stock MATH normalizer for answers such as ``$10\%$``
        # whose gold target is stored as ``10``.
        gold = _parse_expression(_TRAILING_PERCENT_RE.sub("", target))
        prediction = _parse_expression(_TRAILING_PERCENT_RE.sub("", candidate))
        if not gold or not prediction:
            return False, None
        return bool(verify(gold, prediction)), None
    except Exception as exc:  # math_verify must never invalidate the raw bundle.
        return False, f"{type(exc).__name__}: {exc}"


def _sample_path(run_dir: Path) -> Path:
    paths = sorted(run_dir.rglob("samples_hendrycks_math500_*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one MATH-500 sample file under {run_dir}, found {len(paths)}")
    return paths[0]


def score_sample_file(sample_path: Path, *, expected_count: int | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_doc_ids: set[Any] = set()
    with sample_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            if doc_id in seen_doc_ids:
                errors.append(f"duplicate doc_id={doc_id!r}")
            seen_doc_ids.add(doc_id)
            target = row.get("target")
            if not isinstance(target, str) or not target.strip():
                errors.append(f"line {line_number}: missing target")
                target = ""
            response = _first_response(row)
            candidate, extraction = extract_answer_prefix(target, response)
            correct, score_error = equivalent(target, candidate)
            strict = row.get("exact_match")
            strict_correct = bool(strict) if isinstance(strict, (int, float, bool)) else False
            rows.append(
                {
                    "line_number": line_number,
                    "doc_id": doc_id,
                    "target": target,
                    "answer_line": first_answer_line(response),
                    "extracted_answer": candidate,
                    "extraction": extraction,
                    "correct": correct,
                    "stock_exact_match": strict_correct,
                    "score_error": score_error,
                }
            )

    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"row count {len(rows)} != expected {expected_count}")
    lost_strict = [row["doc_id"] for row in rows if row["stock_exact_match"] and not row["correct"]]
    if lost_strict:
        errors.append(f"post-hoc scorer lost stock exact positives: {lost_strict}")
    score_errors = [row for row in rows if row["score_error"]]
    if score_errors:
        errors.append(f"math_verify raised on {len(score_errors)} rows")

    count = len(rows)
    correct_count = sum(bool(row["correct"]) for row in rows)
    strict_count = sum(bool(row["stock_exact_match"]) for row in rows)
    accuracy = correct_count / count if count else 0.0
    stderr = math.sqrt(accuracy * (1.0 - accuracy) / count) if count else None
    return {
        "accepted": not errors,
        "schema_version": SCHEMA_VERSION,
        "scorer": METRIC_NAME,
        "sample_file": str(sample_path),
        "sample_sha256": _sha256(sample_path),
        "expected_count": expected_count,
        "row_count": count,
        "unique_doc_count": len(seen_doc_ids),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "stderr": stderr,
        "stock_exact_correct_count": strict_count,
        "stock_exact_accuracy": strict_count / count if count else 0.0,
        "rescued_count": sum(
            bool(row["correct"]) and not bool(row["stock_exact_match"]) for row in rows
        ),
        "lost_stock_exact_count": len(lost_strict),
        "errors": errors,
        "rows": rows,
    }


def ensure_sidecar(
    run_dir: Path,
    *,
    expected_count: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    sample_path = _sample_path(run_dir)
    output = run_dir / SIDECAR_NAME
    sample_sha = _sha256(sample_path)
    if output.is_file() and not force:
        try:
            cached = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = {}
        if (
            cached.get("schema_version") == SCHEMA_VERSION
            and cached.get("scorer") == METRIC_NAME
            and cached.get("sample_sha256") == sample_sha
            and cached.get("expected_count") == expected_count
            and cached.get("accepted") is True
        ):
            return cached

    report = score_sample_file(sample_path, expected_count=expected_count)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = ensure_sidecar(
        args.run_dir,
        expected_count=args.expected_count,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
