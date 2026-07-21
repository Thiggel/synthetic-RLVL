from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "audit_branchproof_unique_v2_pilot_eval.py"
SPEC = importlib.util.spec_from_file_location("audit_branchproof_unique_v2_pilot_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_artifacts(
    tmp_path: Path,
    *,
    stale_constants: bool = False,
    missing_metric: bool = False,
    missing_nl_metric: bool = False,
    missing_sample_metric: bool = False,
):
    steps = [1, 18]
    k_values = [1, 2]
    metrics = {
        "posthoc/prompts": 4.0,
        "posthoc/sampled_generations_per_prompt": 2.0,
    }
    for step in steps:
        for name in AUDIT.GREEDY_METRICS + AUDIT.NL_GREEDY_METRICS:
            metrics[f"synthetic/step_{step}/{name}"] = 0.5
        for name in AUDIT.SAMPLED_METRICS + AUDIT.NL_SAMPLED_METRICS:
            metrics[f"synthetic_sampled/step_{step}/{name}@1"] = 0.5
            metrics[f"synthetic_sampled/step_{step}/{name}@2"] = 0.75
    if missing_metric:
        del metrics["synthetic/step_18/correct"]
    if missing_nl_metric:
        del metrics["synthetic_sampled/step_18/nl_logic_joint_pass@2"]

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": (
                    "/tmp/branchproof_unique_v2_nl_exact_checkpoint"
                    if missing_nl_metric
                    else "/tmp/branchproof_unique_v2_checkpoint"
                ),
                "profile": "sft",
                "metrics": metrics,
            }
        )
    )
    samples_path = tmp_path / "samples.jsonl"
    rows = []
    for step in steps:
        constants = range(18 if stale_constants and step == 18 else step + 1)
        question = "\n".join(f"c{constant} is blue." for constant in constants)
        rows.append(
            {
                "source": "synthetic",
                "step": step,
                "prompt": f"<question>\n{question}\n</question>",
                "generation": "<formal>proof</formal><answer>yes</answer>",
                "gold_answer": "yes",
                "syntactic": 1.0,
                "format_ok": 1.0,
                "correct": 1.0,
                "valid": 1.0,
                "citation_free_valid": 1.0,
                "citation_free_validity_error": None,
                "citation_free_invalid_line_errors": [],
                "citation_free_line_valid_fraction": 1.0,
                "nl_logic_parse": 0.0,
                "nl_logic_citation_free_valid": 0.0,
            }
        )
    if missing_sample_metric:
        del rows[-1]["nl_logic_citation_free_valid"]
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return metrics_path, samples_path


def _audit(tmp_path: Path, **fixture_kwargs):
    metrics_path, samples_path = _write_artifacts(tmp_path, **fixture_kwargs)
    return AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )


def test_accepts_complete_corrected_artifacts(tmp_path: Path):
    report = _audit(tmp_path)
    assert report["accepted"]
    assert report["samples_audit"]["fresh_constant_failure_count"] == 0


def test_rejects_legacy_wrapped_constants(tmp_path: Path):
    report = _audit(tmp_path, stale_constants=True)
    assert not report["accepted"]
    assert report["samples_audit"]["fresh_constant_failure_count"] == 1
    assert any("expected c0..c18" in error for error in report["errors"])


def test_rejects_missing_primary_metric(tmp_path: Path):
    report = _audit(tmp_path, missing_metric=True)
    assert not report["accepted"]
    assert any("synthetic/step_18/correct" in error for error in report["errors"])


def test_can_allow_all_zero_train_metrics_for_tiny_models(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    payload = json.loads(metrics_path.read_text())
    for step in (1, 18):
        for name in ("syntactic", "format", "correct"):
            payload["metrics"][f"synthetic/step_{step}/{name}"] = 0.0
    metrics_path.write_text(json.dumps(payload))

    default_report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )
    tiny_report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
        require_train_signal=False,
    )

    assert not default_report["accepted"]
    assert tiny_report["accepted"]


def test_nl_train_signal_uses_translation_parse_not_formal_syntax(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    payload = json.loads(metrics_path.read_text())
    payload["checkpoint"] = "/tmp/branchproof_unique_v2_nl_exact_checkpoint"
    for step in (1, 18):
        payload["metrics"][f"synthetic/step_{step}/syntactic"] = 0.0
    metrics_path.write_text(json.dumps(payload))

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )

    assert report["accepted"]


def test_conditioned_nl_filename_uses_translation_metrics(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    conditioned_path = tmp_path / "run_conditioned_nl_passk.json"
    payload = json.loads(metrics_path.read_text())
    for step in (1, 18):
        payload["metrics"][f"synthetic/step_{step}/syntactic"] = 0.0
    conditioned_path.write_text(json.dumps(payload))

    report = AUDIT.audit_artifacts(
        conditioned_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )

    assert report["accepted"]
    assert report["nl_only_surface"]


def test_hybrid_surface_remains_formal_audited(tmp_path: Path):
    assert not AUDIT._is_nl_only_surface(
        "/tmp/branchproof_unique_v2_think_formal_checkpoint",
        tmp_path / "run_think_formal_passk.json",
    )


def test_rejects_missing_translated_nl_metric(tmp_path: Path):
    report = _audit(tmp_path, missing_nl_metric=True)
    assert not report["accepted"]
    assert any("nl_logic_joint_pass@2" in error for error in report["errors"])


def test_rejects_missing_sample_validity_metric(tmp_path: Path):
    report = _audit(tmp_path, missing_sample_metric=True)
    assert not report["accepted"]
    assert any("invalid nl_logic_citation_free_valid" in error for error in report["errors"])


def test_rejects_credited_multi_line_answer(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
    rows[-1]["generation"] = "<formal>proof</formal><answer>no\nyes</answer>"
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )

    assert not report["accepted"]
    assert any("correct=1 with 2 answer lines" in error for error in report["errors"])


def test_rejects_credited_single_line_answer_list(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
    rows[-1]["generation"] = "<formal>proof</formal><answer>no yes</answer>"
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )

    assert not report["accepted"]
    assert any("correct=1 with a nonmatching answer line" in error for error in report["errors"])


def test_rejects_sample_validity_diagnostic_contradiction(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
    rows[-1]["citation_free_validity_error"] = "premise parse failed: malformed"
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )

    assert not report["accepted"]
    assert any("citation_free_valid=1 with error" in error for error in report["errors"])


def test_rejects_credited_duplicate_logic_declaration(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
    rows[-1]["generation"] = """<formal>
<constants>
c0 = c0
c1 = c1
</constants>
<predicates>
Ax: x is amber
Ax: x is violet
</predicates>
<premises>
A(c0)
</premises>
<proof>
A(c0) ; R
</proof>
<conclusion>
A(c0)
</conclusion>
</formal>
<answer>yes</answer>"""
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
    )

    assert not report["accepted"]
    assert report["samples_audit"]["credited_duplicate_declaration_failure_count"] == 1
    assert any("duplicate logic declarations" in error for error in report["errors"])


def test_audits_complete_generation_log_and_records_cap_hits(tmp_path: Path):
    log_path = tmp_path / "eval.out"
    log_path.write_text(
        "\n".join(
            [
                "[syntheval] greedy vLLM chunk 1/1 done in 2.0s "
                "(100 output tokens, max=50)",
                "[syntheval] sampled vLLM chunk 1/2 done in 3.0s "
                "(200 output tokens, max=75)",
                "[syntheval] sampled vLLM chunk 2/2 done in 5.0s "
                "(300 output tokens, max=100)",
            ]
        )
        + "\n"
    )
    errors = []

    report = AUDIT._audit_eval_log(
        log_path,
        expected_greedy_chunks=1,
        expected_sampled_chunks=2,
        generation_cap=100,
        errors=errors,
    )

    assert errors == []
    assert report["greedy"]["tokens_per_second"] == 50.0
    assert report["sampled"]["output_tokens"] == 500
    assert report["sampled"]["max_output_tokens"] == 100
    assert report["sampled"]["cap_hit_chunk_count"] == 1


def test_rejects_incomplete_generation_log(tmp_path: Path):
    log_path = tmp_path / "eval.out"
    log_path.write_text(
        "[syntheval] greedy vLLM chunk 1/1 done in 2.0s "
        "(100 output tokens, max=50)\n"
    )
    errors = []

    AUDIT._audit_eval_log(
        log_path,
        expected_greedy_chunks=1,
        expected_sampled_chunks=2,
        generation_cap=100,
        errors=errors,
    )

    assert any("sampled completed chunk indices=[]" in error for error in errors)


def test_accepts_complete_sampled_qualitative_artifacts(tmp_path: Path):
    steps = [1, 25]
    metrics = {
        "posthoc/prompts": 4.0,
        "posthoc/sampled_generations_per_prompt": 2.0,
    }
    for step in steps:
        for name in AUDIT.SAMPLED_METRICS:
            metrics[f"synthetic_sampled/step_{step}/{name}@1"] = 0.5
            metrics[f"synthetic_sampled/step_{step}/{name}@2"] = 0.75
    metrics_path = tmp_path / "sampled_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": "/tmp/branchproof_unique_v2_checkpoint",
                "profile": "sft",
                "metrics": metrics,
            }
        )
    )

    rows = []
    for step in steps:
        question = "\n".join(f"c{constant} is blue." for constant in range(step + 1))
        for prompt_index in range(2):
            for sample_index in range(2):
                rows.append(
                    {
                        "source": "synthetic_sampled",
                        "step": step,
                        "prompt": f"<question>\n{question}\nitem {prompt_index}\n</question>",
                        "generation": "<formal>proof</formal><answer>yes</answer>",
                        "gold_answer": "yes",
                        "sample_index": sample_index,
                        "syntactic": 1.0,
                        "format_ok": 1.0,
                        "correct": 1.0,
                        "valid": 1.0,
                        "citation_free_valid": 1.0,
                        "citation_free_validity_error": None,
                        "citation_free_invalid_line_errors": [],
                        "citation_free_line_valid_fraction": 1.0,
                        "nl_logic_parse": 0.0,
                        "nl_logic_citation_free_valid": 0.0,
                    }
                )
    samples_path = tmp_path / "sampled_samples.jsonl"
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=steps,
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=8,
        train_max=25,
        greedy_required=False,
        expected_sample_source="synthetic_sampled",
        expected_samples_per_step=4,
        expected_sample_indices=[0, 1],
    )

    assert report["accepted"]
    assert report["samples_audit"]["unique_prompt_counts"] == {"1": 2, "25": 2}
    assert report["samples_audit"]["sample_index_counts"] == {
        "1": {0: 2, 1: 2},
        "25": {0: 2, 1: 2},
    }


def test_filters_combined_greedy_and_sampled_retention(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    sampled_rows = []
    for step in [1, 18]:
        question = "\n".join(f"c{constant} is blue." for constant in range(step + 1))
        for prompt_index in range(2):
            for sample_index in range(2):
                sampled_rows.append(
                    {
                        "source": "synthetic_sampled",
                        "step": step,
                        "prompt": f"<question>\n{question}\nitem {prompt_index}\n</question>",
                        "generation": "<formal>proof</formal><answer>yes</answer>",
                        "gold_answer": "yes",
                        "sample_index": sample_index,
                        "syntactic": 1.0,
                        "format_ok": 1.0,
                        "correct": 1.0,
                        "valid": 1.0,
                        "citation_free_valid": 1.0,
                        "citation_free_validity_error": None,
                        "citation_free_invalid_line_errors": [],
                        "citation_free_line_valid_fraction": 1.0,
                        "nl_logic_parse": 0.0,
                        "nl_logic_citation_free_valid": 0.0,
                    }
                )
    with samples_path.open("a", encoding="utf-8") as handle:
        for row in sampled_rows:
            handle.write(json.dumps(row) + "\n")

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=8,
        train_max=18,
        expected_sample_source="synthetic_sampled",
        sample_source_filter="synthetic_sampled",
        expected_total_samples=10,
        expected_samples_per_step=4,
        expected_unique_prompts_per_step=2,
        expected_sample_indices=[0, 1],
    )

    assert report["accepted"]
    assert report["total_sample_count"] == 10
    assert report["samples_audit"]["sample_count"] == 8


def test_rejects_incomplete_prompt_coverage_after_source_filter(tmp_path: Path):
    metrics_path, samples_path = _write_artifacts(tmp_path)
    rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
    for row in rows:
        row["source"] = "synthetic_sampled"
        row["sample_index"] = 0
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = AUDIT.audit_artifacts(
        metrics_path,
        samples_path,
        steps=[1, 18],
        k_values=[1, 2],
        samples_per_step=2,
        generations_per_prompt=2,
        expected_retained_samples=2,
        train_max=18,
        expected_sample_source="synthetic_sampled",
        sample_source_filter="synthetic_sampled",
        expected_total_samples=2,
        expected_unique_prompts_per_step=2,
        expected_sample_indices=[0],
    )

    assert not report["accepted"]
    assert any("unique prompts=1, expected 2" in error for error in report["errors"])
