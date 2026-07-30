from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "rescore_math500.py"
SPEC = importlib.util.spec_from_file_location("rescore_math500", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RESCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESCORE)


def _row(doc_id: int, target: str, response: str, *, strict: int = 0) -> dict:
    return {
        "doc_id": doc_id,
        "doc": {"answer": target},
        "target": target,
        "filtered_resps": [response],
        "filter": "none",
        "exact_match": strict,
    }


def test_answer_prefix_accepts_equivalent_answer_before_explanation() -> None:
    candidate, reason = RESCORE.extract_answer_prefix("-50", "$-50$.\nSolution: -50")
    assert candidate == "-50"
    assert reason == "leading_math"
    assert RESCORE.equivalent("-50", candidate) == (True, None)


def test_answer_prefix_accepts_stock_normalized_percent_answer() -> None:
    candidate, reason = RESCORE.extract_answer_prefix("10", r"$10\%$")
    assert candidate == r"10\%"
    assert reason == "leading_math"
    assert RESCORE.equivalent("10", candidate) == (True, None)


def test_answer_prefix_ignores_escaped_currency_dollar_inside_math() -> None:
    candidate, reason = RESCORE.extract_answer_prefix(
        r"\$36",
        r"The original price of the shirt was $\$36$.",
    )
    assert candidate == r"\$36"
    assert reason == "last_explicit_token"
    assert RESCORE.equivalent(r"\$36", candidate) == (True, None)


def test_answer_prefix_accepts_final_rhs_of_direct_equation() -> None:
    candidate, _ = RESCORE.extract_answer_prefix("6+9i", "$6+12i-3i=6+9i$.")
    assert candidate == "6+9i"
    assert RESCORE.equivalent("6+9i", candidate) == (True, None)


def test_answer_prefix_rejects_gold_in_later_prompt_repetition() -> None:
    candidate, _ = RESCORE.extract_answer_prefix("4", "10\nQuestion: the answer is 4")
    assert candidate == "10"
    assert RESCORE.equivalent("4", candidate) == (False, None)


def test_answer_prefix_rejects_gold_inside_wrong_leading_explanation() -> None:
    response = "13. The factors are 1, 2, 7, 14, 21, 42, and 84."
    candidate, _ = RESCORE.extract_answer_prefix("14", response)
    assert candidate == "13"
    assert RESCORE.equivalent("14", candidate) == (False, None)


def test_answer_prefix_falls_back_to_final_explicit_answer_after_prose() -> None:
    response = (
        "To solve the equation, cross-multiply.\n"
        r"Therefore, the solution is $x = 11$."
    )
    candidate, reason = RESCORE.extract_answer_prefix("11", response)
    assert candidate == "11"
    assert reason == "final_explicit_token"
    assert RESCORE.equivalent("11", candidate) == (True, None)


def test_answer_prefix_fallback_rejects_next_prompt_gold() -> None:
    response = "I cannot solve this.\nQuestion: a new problem whose answer is 11"
    candidate, reason = RESCORE.extract_answer_prefix("11", response)
    assert candidate is None
    assert reason == "no_answer_token"


def test_answer_prefix_prefers_explicit_math_answer_over_malformed_suffix() -> None:
    response = (
        "To solve the equation, cross-multiply.\n"
        r"Therefore, the solution is $x = 11$." "\n"
        "The answer is: 11.00000000000007.00000000000007.00"
    )
    candidate, reason = RESCORE.extract_answer_prefix("11", response)
    assert candidate == "11"
    assert reason == "final_explicit_token"
    assert RESCORE.equivalent("11", candidate) == (True, None)


def test_answer_prefix_rejects_extra_comma_separated_value() -> None:
    candidate, reason = RESCORE.extract_answer_prefix("1", "1, 2\nSolution: ...")
    assert candidate == "1, 2"
    assert reason == "leading_number_list"
    assert RESCORE.equivalent("1", candidate) == (False, None)


def test_full_equation_and_tuple_do_not_collapse_to_shared_numbers() -> None:
    equation, _ = RESCORE.extract_answer_prefix(
        "5x - 7y + 11z + 4 = 0",
        "$x - 2y + 2z - 4 = 0.$",
    )
    assert RESCORE.equivalent("5x - 7y + 11z + 4 = 0", equation) == (False, None)
    coordinates, _ = RESCORE.extract_answer_prefix(
        r"\left( \frac{3}{2}, -13 \right)",
        "(2/3, -13/3)",
    )
    assert RESCORE.equivalent(r"\left( \frac{3}{2}, -13 \right)", coordinates) == (
        False,
        None,
    )


def test_incomplete_repetition_is_not_scored() -> None:
    candidate, reason = RESCORE.extract_answer_prefix("5", "$5^5=5^4+")
    assert candidate is None
    assert reason == "unbalanced_math_delimiter"


def test_sample_report_preserves_strict_positives_and_writes_reproducible_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples_hendrycks_math500_2026.jsonl"
    rows = [
        _row(0, "-50", "$-50$.\nSolution: -50"),
        _row(1, "6+9i", "$6+12i-3i=6+9i$."),
        _row(2, "4", "10\nQuestion: answer 4"),
        _row(3, r"11\sqrt2", r"$11\sqrt2$.", strict=1),
        _row(4, "1", "1, 2\nSolution: ..."),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = RESCORE.score_sample_file(path, expected_count=5)
    assert report["accepted"]
    assert report["correct_count"] == 3
    assert report["stock_exact_correct_count"] == 1
    assert report["rescued_count"] == 2
    assert report["lost_stock_exact_count"] == 0
