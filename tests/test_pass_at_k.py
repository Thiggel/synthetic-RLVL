from __future__ import annotations

import pytest

from synthrlvl.evaluation.pass_at_k import pass_at_k_estimate, score_pass_at_k
from synthrlvl.metrics import OutputEvaluator
from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def _task_cfg() -> TaskConfig:
    return TaskConfig(
        template=TemplateName.LOGIC,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(1, 2),
        val_steps=StepRange(1, 4),
        seed=42,
    )


def test_pass_at_k_estimator_boundaries():
    assert pass_at_k_estimate(4, 0, 1) == 0.0
    assert pass_at_k_estimate(4, 1, 4) == 1.0
    assert pass_at_k_estimate(4, 2, 2) == pytest.approx(1.0 - 1.0 / 6.0)
    with pytest.raises(ValueError):
        pass_at_k_estimate(2, 1, 4)


def test_score_pass_at_k_joint_and_valid_given_correct():
    cfg = _task_cfg()
    sample = TaskBuilder(cfg).sample(0, train=False)

    correct_valid = sample.target
    correct_invalid = sample.target.replace("<proof>", "<proof>\nAa ; ->E,999,998\n", 1)
    wrong_invalid = "<formal></formal><answer>definitely_wrong</answer>"

    metrics = score_pass_at_k(
        records=[
            type(
                "Rec",
                (),
                {
                    "step": 3,
                    "template": cfg.template,
                    "gold_answer": sample.answer,
                    "gold_logic_premises": sample.logic_premises,
                    "gold_logic_conclusion": sample.logic_conclusion,
                    "prefill": cfg.prefill,
                    "gold_first_modality_lines": sample.gold_first_modality_lines,
                },
            )()
        ],
        generations_by_record=[[correct_valid, correct_invalid, wrong_invalid, wrong_invalid]],
        output_eval=OutputEvaluator(),
        k_values=[1, 2, 4],
        band_predicates={"ood": lambda step: step > 2},
    )

    assert metrics["synthetic_sampled/step_3/correct_pass@4"] == 1.0
    assert metrics["synthetic_sampled/step_3/joint_pass@4"] == 1.0
    assert metrics["synthetic_sampled/step_3/valid_given_correct@1"] == pytest.approx(0.5)
    assert metrics["synthetic_sampled/band_ood/correct_pass@4"] == 1.0

