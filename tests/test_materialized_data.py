from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from synthrlvl.datasets import MaterializedSyntheticDataset
from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def _task_cfg() -> TaskConfig:
    return TaskConfig(
        template=TemplateName.LOGIC,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(1, 5),
        val_steps=StepRange(1, 10),
        seed=3407,
    )


def test_load_materialized_rows_local(tmp_path: Path):
    mat_ds = MaterializedSyntheticDataset()
    root = tmp_path / "materialized"
    subset = mat_ds.train_up_to_5_subset
    pq_file = root / subset / "train.parquet"
    pq_file.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(
        [
            {
                "depth": 1,
                "question_fol": "q",
                "question_nl": "q",
                "answer": "a",
                "constants": [],
                "predicates": [],
                "premises_fol": [],
                "premises_nl": [],
                "proof_fol": ["1. Aa ; R,1"],
                "proof_nl": ["1. x"],
                "metadata": {"depth": 1},
                "record_index": 0,
            }
        ]
    ).to_parquet(str(pq_file))

    loaded = mat_ds.load_rows(subset=subset, local_root=str(root), limit=1)
    assert len(loaded) == 1
    assert int(loaded[0]["depth"]) == 1
    assert mat_ds.materialized_parquet_path(str(root), subset) == pq_file.resolve()


def test_task_sample_from_materialized_row():
    row = {
        "depth": 1,
        "record_index": 0,
        "question_fol": "What value of color does a have?",
        "question_nl": "What color does Gary have?",
        "answer": "blue",
        "constants": ["1. a = Gary"],
        "predicates": ["1. Ax: x is blue"],
        "premises_fol": ["1. Aa"],
        "premises_nl": ["1. Gary is blue."],
        "proof_fol": ["2. Aa ; R,1"],
        "proof_nl": ["2. Gary is blue."],
        "metadata": {"depth": 1},
    }
    sample = task_sample_from_materialized_row(row, cfg=_task_cfg())
    assert "<formal>" in sample.target
    assert "<answer>" in sample.target
    assert sample.answer == "blue"
