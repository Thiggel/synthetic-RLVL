from __future__ import annotations

import pytest

from logic_engine import LogicEngine
from synthetic_dataset import DatasetConfig, LogicDatasetGenerator


def _extract_conclusion(proof_line: str) -> str:
    return proof_line.split(". ", 1)[1].split(" ; ", 1)[0].strip()


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("index", [0, 1, 3])
def test_generated_proof_is_valid_for_multiple_depths(depth: int, index: int):
    cfg = DatasetConfig(depth=depth, distractor_ratio=0.5, seed=13)
    example = LogicDatasetGenerator(cfg).generate(index)

    engine = LogicEngine()
    report = engine.analyze_proof(
        premises="\n".join(example.premises_fol),
        conclusion=_extract_conclusion(example.proof_fol[-1]),
        proof="\n".join(example.proof_fol),
    )

    assert report.ok is True
    assert report.conclusion_supported is True
    assert all(line.valid for line in report.lines)


@pytest.mark.parametrize("depth", [1, 2, 4, 7])
def test_distractor_count_matches_ratio(depth: int):
    cfg = DatasetConfig(depth=depth, distractor_ratio=0.5, seed=7)
    gen = LogicDatasetGenerator(cfg)
    ex = gen.generate(0)
    expected = round(gen._num_support_premises() * cfg.distractor_ratio)
    assert ex.metadata["num_distractors"] == expected


def test_zero_distractors_supported():
    cfg = DatasetConfig(depth=4, distractor_ratio=0.0, seed=11)
    ex = LogicDatasetGenerator(cfg).generate(0)
    assert ex.metadata["num_distractors"] == 0


def test_generation_is_deterministic():
    cfg = DatasetConfig(depth=6, distractor_ratio=0.5, seed=99)
    gen = LogicDatasetGenerator(cfg)
    ex_a = gen.generate(3).to_dict()
    ex_b = gen.generate(3).to_dict()
    assert ex_a == ex_b
