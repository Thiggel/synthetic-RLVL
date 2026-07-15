from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "aggregate_nanotron_multihop_eval.py"
SPEC = importlib.util.spec_from_file_location("aggregate_nanotron_multihop_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summary_is_ordered_control_nl_logic_and_averages_benchmarks() -> None:
    values = {"control": 0.3, "nl_p15": 0.5, "logic_p15": 0.7}
    rows = []
    for condition, value in values.items():
        for mode in ("direct", "instruction"):
            for protocol in ("standard_short_answer", "strict_tagged"):
                for benchmark in ("hotpotqa", "2wikimqa", "musique"):
                    rows.append(
                        {
                            "condition": condition,
                            "mode": mode,
                            "protocol": protocol,
                            "benchmark": benchmark,
                            "qa_f1": value,
                            "qa_exact_match": value if protocol == "strict_tagged" else None,
                            "tag_found": 1.0 if protocol == "strict_tagged" else None,
                        }
                    )

    summary = MODULE.summarize_rows(rows)

    assert [row["condition"] for row in summary[::4]] == ["control", "nl_p15", "logic_p15"]
    assert len(summary) == 12
    logic_tagged = next(
        row
        for row in summary
        if row["condition"] == "logic_p15"
        and row["mode"] == "direct"
        and row["protocol"] == "strict_tagged"
    )
    assert logic_tagged["benchmark_count"] == 3
    assert abs(logic_tagged["mean_qa_f1"] - 0.7) < 1e-12
    assert abs(logic_tagged["mean_exact_match"] - 0.7) < 1e-12
    assert logic_tagged["mean_tag_found"] == 1.0


def test_summary_rejects_missing_condition_branch_protocol() -> None:
    try:
        MODULE.summarize_rows([])
    except ValueError as exc:
        assert "missing rows" in str(exc)
    else:
        raise AssertionError("incomplete aggregate rows were accepted")


def test_answer_head_rescore_removes_generated_document_continuation() -> None:
    rows = [
        {
            "doc": {"answers": ["Miller v. California"]},
            "resps": [["Miller v. California Passage 1: unrelated continuation"]],
            "qa_f1_score": 0.5,
        },
        {
            "doc": {"answers": ["Ozalj"]},
            "resps": [["wrong answer"]],
            "qa_f1_score": 0.0,
        },
    ]

    result = MODULE.standard_answer_head_rescore(rows)

    assert result["row_count"] == 2
    assert result["stock_qa_f1"] == 0.25
    assert result["answer_head_qa_f1"] == 0.5
    assert result["answer_head_exact_match"] == 0.5


def test_generation_diagnostics_detect_learned_trace_surface() -> None:
    rows = [
        {"resps": [["<formal> proof"]]},
        {"resps": [["<formal> proof You are an AI assistant"]]},
    ]

    result = MODULE.generation_diagnostics(rows)

    assert result["row_count"] == 2
    assert result["formal_open_rate"] == 1.0
    assert result["next_document_marker_rate"] == 0.5
    assert result["think_open_rate"] == 0.0
