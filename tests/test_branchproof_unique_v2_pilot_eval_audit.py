from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "audit_branchproof_unique_v2_pilot_eval.py"
SPEC = importlib.util.spec_from_file_location("audit_branchproof_unique_v2_pilot_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_artifacts(tmp_path: Path, *, stale_constants: bool = False, missing_metric: bool = False):
    steps = [1, 18]
    k_values = [1, 2]
    metrics = {
        "posthoc/prompts": 4.0,
        "posthoc/sampled_generations_per_prompt": 2.0,
    }
    for step in steps:
        for name in AUDIT.GREEDY_METRICS:
            metrics[f"synthetic/step_{step}/{name}"] = 0.5
        for name in AUDIT.SAMPLED_METRICS:
            metrics[f"synthetic_sampled/step_{step}/{name}@1"] = 0.5
            metrics[f"synthetic_sampled/step_{step}/{name}@2"] = 0.75
    if missing_metric:
        del metrics["synthetic/step_18/correct"]

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": "/tmp/branchproof_unique_v2_checkpoint",
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
            }
        )
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
