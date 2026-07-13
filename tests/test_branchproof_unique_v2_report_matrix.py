from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "experiments"
    / "branchproof_unique_v2_report_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("branchproof_unique_v2_report_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_report_matrix_has_expected_three_seed_coverage() -> None:
    expected = {
        "surface": (27, 27),
        "shortcut": (42, 42),
        "hybrid": (30, 30),
        "conditioned10k": (15, 30),
        "conditioned50k": (15, 30),
        "architecture": (54, 54),
        "batch": (36, 48),
        "large": (15, 18),
    }
    for group, (train_count, eval_count) in expected.items():
        train = MODULE.train_rows(group)
        eval_ = MODULE.eval_rows(group)
        assert len(train) == train_count
        assert len(eval_) == eval_count
        assert {row.seed for row in train} == {3407, 3408, 3409}
        assert len({row.run_name for row in train}) == len(train)
        assert all(row.train_index < len(train) for row in eval_)


def test_conditioned_and_batch_rows_expand_to_both_eval_modes() -> None:
    for group in ("conditioned10k", "conditioned50k"):
        assert all(
            row.train.eval_templates == ("conditioned_logic", "conditioned_nl")
            for row in MODULE.eval_rows(group)
        )
    assert all(row.max_steps == 50_000 for row in MODULE.train_rows("conditioned50k"))
    conditioned_batch = [
        row for row in MODULE.train_rows("batch") if row.train_template == "conditioned_dual"
    ]
    assert len(conditioned_batch) == 12
    assert all(row.balanced_modality_batches for row in conditioned_batch)
    assert all(row.per_device_batch_size * row.grad_accum in {2, 4, 8, 16} for row in conditioned_batch)


def test_hybrid_rows_preserve_both_modalities_without_training_truncation() -> None:
    assert all(row.max_length == 16384 for row in MODULE.train_rows("hybrid"))


def test_same_token_formal_reuses_corrected_main_baseline() -> None:
    surface = MODULE.train_rows("surface")
    assert not any("same_target_tokens_logic" in row.run_name for row in surface)
    matched_nl = [row for row in surface if "same_target_tokens_nl_exact" in row.run_name]
    assert len(matched_nl) == 3
    assert all(row.max_steps == 7140 for row in matched_nl)


def test_shortcut_rows_use_correct_neutral_eval_dataset_keys() -> None:
    rows = MODULE.train_rows("shortcut")
    assert {row.data_kind for row in rows} == {
        "shortcut_schema_0p3",
        "shortcut_schema_0p5",
        "shortcut_schema_0p8",
        "shortcut_position_0p5",
        "shortcut_position_0p8",
        "shortcut_initial_marker_0p5",
        "shortcut_initial_marker_0p8",
    }
    assert all(row.train_max == 25 for row in rows)
