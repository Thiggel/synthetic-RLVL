from __future__ import annotations

from synthrlvl.metrics import OutputEvaluator, RewardComputer
from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, RewardSchema, StepRange, TaskConfig, TemplateName


def make_task(template: TemplateName = TemplateName.LOGIC, prefill: PrefillMode = PrefillMode.NONE) -> TaskConfig:
    return TaskConfig(
        template=template,
        prefill=prefill,
        distractor_ratio=0.5,
        train_steps=StepRange(1, 3),
        val_steps=StepRange(1, 3),
        seed=42,
    )


def test_task_builder_emits_tagged_targets():
    builder = TaskBuilder(make_task())
    sample = builder.sample(0, train=True)
    assert sample.prompt
    assert "<logic>" in sample.target
    assert "<answer>" in sample.target


def test_output_evaluator_positive_path_logic():
    cfg = make_task()
    builder = TaskBuilder(cfg)
    sample = builder.sample(1, train=False)
    evaluator = OutputEvaluator()

    result = evaluator.evaluate(
        sample.target,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert result.format_ok == 1.0
    assert result.correct == 1.0
    assert result.valid == 1.0


def test_reward_schema_indicator_all():
    cfg = make_task()
    builder = TaskBuilder(cfg)
    sample = builder.sample(2, train=False)
    rewarder = RewardComputer(OutputEvaluator())

    reward, parts = rewarder.reward(
        sample.target,
        schema=RewardSchema.INDICATOR_ALL,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert reward == 1.0
    assert parts.valid == 1.0


def test_natural_template_has_format_and_answer():
    cfg = make_task(template=TemplateName.NATURAL)
    builder = TaskBuilder(cfg)
    sample = builder.sample(3, train=False)
    result = OutputEvaluator().evaluate(
        sample.target,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert result.format_ok == 1.0
    assert result.correct == 1.0


def test_logic_format_rejects_unexpected_text_outside_tags():
    cfg = make_task(template=TemplateName.LOGIC)
    builder = TaskBuilder(cfg)
    sample = builder.sample(4, train=False)
    bad = sample.target + "\nunexpected trailing text"
    result = OutputEvaluator().evaluate(
        bad,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert result.format_ok == 0.0


def test_logic_valid_fallback_from_headings_without_tags():
    cfg = make_task(template=TemplateName.LOGIC)
    builder = TaskBuilder(cfg)
    sample = builder.sample(5, train=False)

    formal = sample.target.split("<answer>", 1)[0]
    premises = formal.split("<premises>", 1)[1].split("</premises>", 1)[0].strip()
    proof = formal.split("<proof>", 1)[1].split("</proof>", 1)[0].strip()
    conclusion = formal.split("<conclusion>", 1)[1].split("</conclusion>", 1)[0].strip()
    answer = sample.target.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    heading_output = (
        f"Premises:\n{premises}\n\n"
        f"Proof:\n{proof}\n\n"
        f"Conclusion:\n{conclusion}\n\n"
        f"<answer>{answer}</answer>"
    )

    result = OutputEvaluator().evaluate(
        heading_output,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert result.format_ok == 0.0
    assert result.correct == 1.0
    assert result.valid == 1.0
