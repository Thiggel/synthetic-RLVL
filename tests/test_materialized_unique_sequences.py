from __future__ import annotations

from synthetic_dataset import MaterializedDatasetBuilder, MaterializedDatasetSpec


def _records(*, require_unique_sequences: bool) -> list[dict]:
    builder = MaterializedDatasetBuilder()
    builder._core_record = lambda gen, index: {  # type: ignore[method-assign]
        "depth": gen.config.depth,
        "record_index": index,
        "payload": index // 2,
    }
    return list(
        builder._records_for_spec(
            spec=MaterializedDatasetSpec(
                subset="test",
                min_depth=1,
                max_depth=1,
                rows=3,
                seed=3407,
            ),
            distractor_ratio=0.0,
            require_unique_sequences=require_unique_sequences,
        )
    )


def test_materializer_can_skip_duplicate_rendered_sequences() -> None:
    rows = _records(require_unique_sequences=True)

    assert [row["record_index"] for row in rows] == [0, 2, 4]
    assert [row["payload"] for row in rows] == [0, 1, 2]


def test_materializer_preserves_legacy_fixed_index_behavior_by_default() -> None:
    rows = _records(require_unique_sequences=False)

    assert [row["record_index"] for row in rows] == [0, 1, 2]
    assert [row["payload"] for row in rows] == [0, 0, 1]


def test_materializer_rejects_a_duplicate_in_either_training_modality() -> None:
    builder = MaterializedDatasetBuilder()

    def record(gen, index: int) -> dict:
        return {
            "depth": gen.config.depth,
            "record_index": index,
            "constants": ["a = Ada"],
            "predicates": ["Ax: state"],
            "premises_fol": [f"1. A{index // 2}"],
            "proof_fol": [f"2. A{index // 2};R,P1"],
            "question_fol": "Which state?",
            "premises_nl": [f"1. NL premise {index}"],
            "proof_nl": [f"2. NL proof {index}"],
            "question_nl": "Which state?",
            "answer": "state",
        }

    builder._core_record = record  # type: ignore[method-assign]
    rows = list(
        builder._records_for_spec(
            spec=MaterializedDatasetSpec(
                subset="test",
                min_depth=1,
                max_depth=1,
                rows=3,
                seed=3407,
            ),
            distractor_ratio=0.0,
            require_unique_sequences=True,
        )
    )

    assert [row["record_index"] for row in rows] == [0, 2, 4]
