from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analysis"
    / "audit_branchproof_unique_v2_qualitative_grid.py"
)
SPEC = importlib.util.spec_from_file_location("audit_branchproof_qualitative", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sample(step: int, sample_index: int) -> dict[str, object]:
    correct = float(sample_index == 0)
    return {
        "source": "synthetic_sampled",
        "step": step,
        "sample_index": sample_index,
        "gold_answer": "willow",
        "generation": f"<formal>depth {step}</formal><answer>willow</answer>",
        "correct": correct,
        "format_ok": 1.0,
        "citation_free_valid": correct,
        "nl_logic_citation_free_valid": correct,
    }


def _write_grid(root: Path, logs: Path, *, omit_last: bool = False) -> None:
    keys = [
        (template, train_max, seed)
        for template in MODULE.TEMPLATES
        for train_max in MODULE.TRAIN_MAXES
        for seed in MODULE.SEEDS
    ]
    if omit_last:
        keys.pop()
    for template, train_max, seed in keys:
        name = (
            f"sft_branchproof_unique_v2_{template}_train1to{train_max}_10k_"
            f"seed{seed}_samples.jsonl"
        )
        rows = [_sample(depth, sample_index) for depth in MODULE.DEPTHS for sample_index in (0, 1)]
        (root / name).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        index = MODULE._array_index(template, train_max, seed)
        (logs / f"eval_bp_unique_123_{index}.out").write_text(
            "[syntheval] sampled vLLM chunk 1/4 done in 1.0s "
            "(10 output tokens, max=7168)\n",
            encoding="utf-8",
        )


def test_audits_complete_grid_and_selects_cases(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    log_dir = tmp_path / "logs"
    final_dir.mkdir()
    log_dir.mkdir()
    _write_grid(final_dir, log_dir)

    report = MODULE.audit_grid(
        final_dir,
        log_dir,
        eval_array_job_id="123",
        expected_sampled_rows=len(MODULE.DEPTHS) * 2,
        expected_rows_per_depth=2,
        generation_cap=7168,
    )

    assert report["accepted"] is True
    assert report["observed_grid_size"] == 30
    assert report["coverage"]["logic:correct"] == 15
    assert report["coverage"]["nl_exact:incorrect"] == 15
    first = report["runs"][0]
    assert set(first["slice_selections"]) == {"shallow", "train_edge", "ood_edge", "depth50"}
    assert first["cap_hit_chunks"] == [1]


def test_rejects_missing_grid_row(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    log_dir = tmp_path / "logs"
    final_dir.mkdir()
    log_dir.mkdir()
    _write_grid(final_dir, log_dir, omit_last=True)

    report = MODULE.audit_grid(
        final_dir,
        log_dir,
        eval_array_job_id="123",
        expected_sampled_rows=len(MODULE.DEPTHS) * 2,
        expected_rows_per_depth=2,
        generation_cap=7168,
    )

    assert report["accepted"] is False
    assert any("missing sample artifact" in error for error in report["errors"])


def test_rejects_missing_modality_validity_metric(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    log_dir = tmp_path / "logs"
    final_dir.mkdir()
    log_dir.mkdir()
    _write_grid(final_dir, log_dir)
    path = next(final_dir.glob("*nl_exact*_samples.jsonl"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0].pop("nl_logic_citation_free_valid")
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = MODULE.audit_grid(
        final_dir,
        log_dir,
        eval_array_job_id="123",
        expected_sampled_rows=len(MODULE.DEPTHS) * 2,
        expected_rows_per_depth=2,
        generation_cap=7168,
    )

    assert report["accepted"] is False
    assert any("invalid nl_logic_citation_free_valid" in error for error in report["errors"])
