from __future__ import annotations

import pytest

from synthetic_dataset import DatasetConfig, LogicDatasetGenerator
from synthrlvl.metrics import OutputEvaluator, RewardComputer
from synthrlvl.task import TaskBuilder, task_sample_from_logic_example
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
    assert "<formal>" in sample.target
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


def test_reward_schema_line_valid_fraction_partial_credit():
    cfg = make_task()
    builder = TaskBuilder(cfg)
    sample = builder.sample(6, train=False)
    rewarder = RewardComputer(OutputEvaluator())

    formal = sample.target.split("<answer>", 1)[0]
    proof = formal.split("<proof>", 1)[1].split("</proof>", 1)[0].strip()
    proof_lines = [ln for ln in proof.splitlines() if ln.strip()]
    assert len(proof_lines) >= 2
    # Force one clearly invalid line via forward citation while keeping the rest intact.
    proof_lines[0] = "Aa ; ->E,2,999"
    bad_target = sample.target.replace(proof, "\n".join(proof_lines), 1)

    reward, parts = rewarder.reward(
        bad_target,
        schema=RewardSchema.CORRECT_PLUS_LINE_VALID_PLUS_0P1_FORMAT,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    # Reward must include partial line-valid credit, strictly between correctness-only
    # and correctness+full-validity for this corrupted proof.
    assert parts.correct == 1.0
    assert reward > 1.0
    assert reward < 2.1


def test_citation_free_validity_reward_accepts_no_citation_gold_trace():
    cfg = make_task()
    cfg = TaskConfig(
        template=cfg.template,
        prefill=cfg.prefill,
        distractor_ratio=cfg.distractor_ratio,
        train_steps=cfg.train_steps,
        val_steps=cfg.val_steps,
        seed=cfg.seed,
        difficulty="hard_v5",
        shortcut_rate=0.8,
    )
    sample = TaskBuilder(cfg).sample(0, train=False)
    rewarder = RewardComputer(OutputEvaluator())

    reward, parts = rewarder.reward(
        sample.target,
        schema=RewardSchema.CORRECT_PLUS_CITATION_FREE_VALID_PLUS_0P1_FORMAT,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )

    assert parts.valid == 0.0
    assert parts.citation_free_valid == 1.0


def test_reward_schema_indicator_correct_and_citation_free_valid_plus_format():
    cfg = DatasetConfig(depth=2, difficulty="hard_fsa_schema", branching_factor=4, shortcut_rate=1.0, seed=3407)
    ex = LogicDatasetGenerator(cfg).generate(0)
    task_cfg = TaskConfig(
        template=TemplateName.LOGIC,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(1, 2),
        val_steps=StepRange(1, 2),
        seed=3407,
        difficulty="hard_fsa_schema",
        branching_factor=4,
        shortcut_rate=1.0,
    )
    sample = task_sample_from_logic_example(ex, cfg=task_cfg, depth=2)
    evaluator = OutputEvaluator()
    reward = RewardComputer(evaluator)

    value, parts = reward.reward(
        sample.target,
        schema=RewardSchema.INDICATOR_CORRECT_AND_CITATION_FREE_VALID_PLUS_0P1_FORMAT,
        template=TemplateName.LOGIC,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=PrefillMode.NONE,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )

    assert parts.correct == 1.0
    assert parts.citation_free_valid == 1.0
    assert value == pytest.approx(1.1)


def test_reward_schema_line_valid_plus_correct_if_full_valid_gates_correctness():
    cfg = DatasetConfig(depth=3, difficulty="hard_fsa_schema", branching_factor=4, shortcut_rate=1.0, seed=3407)
    ex = LogicDatasetGenerator(cfg).generate(0)
    task_cfg = TaskConfig(
        template=TemplateName.LOGIC,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(1, 3),
        val_steps=StepRange(1, 3),
        seed=3407,
        difficulty="hard_fsa_schema",
        branching_factor=4,
        shortcut_rate=1.0,
    )
    sample = task_sample_from_logic_example(ex, cfg=task_cfg, depth=3)
    reward = RewardComputer(OutputEvaluator())

    full_value, full_parts = reward.reward(
        sample.target,
        schema=RewardSchema.CITATION_FREE_LINE_VALID_PLUS_CORRECT_IF_FULL_VALID_PLUS_0P1_FORMAT,
        template=TemplateName.LOGIC,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=PrefillMode.NONE,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )

    assert full_parts.correct == 1.0
    assert full_parts.citation_free_valid == 1.0
    assert full_value == pytest.approx(2.1)

    proof = sample.target.split("<proof>", 1)[1].split("</proof>", 1)[0].strip()
    proof_lines = [line for line in proof.splitlines() if line.strip()]
    assert len(proof_lines) >= 2
    incomplete_target = sample.target.replace(proof, proof_lines[0], 1)
    assert reward._line_valid_fraction(incomplete_target, template=TemplateName.LOGIC, citation_free=True) == 1.0

    gated_value, gated_parts = reward.reward(
        incomplete_target,
        schema=RewardSchema.CITATION_FREE_LINE_VALID_PLUS_CORRECT_IF_FULL_VALID_PLUS_0P1_FORMAT,
        template=TemplateName.LOGIC,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=PrefillMode.NONE,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )

    assert gated_parts.correct == 1.0
    assert gated_parts.citation_free_valid == 0.0
    assert gated_value == pytest.approx(1.1)


def test_task_builder_keeps_hard_fsa_schema_eval_shortcut_neutral():
    cfg = TaskConfig(
        template=TemplateName.LOGIC,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(1, 1),
        val_steps=StepRange(1, 1),
        seed=3407,
        difficulty="hard_fsa_schema",
        branching_factor=4,
        shortcut_rate=1.0,
    )
    builder = TaskBuilder(cfg)

    train_sample = builder.sample(0, train=True)
    eval_sample = builder.sample(0, train=False)

    assert train_sample.metadata["shortcut_enabled"] is True
    assert eval_sample.metadata["shortcut_enabled"] is False
    assert eval_sample.metadata["split_intervention"] == "eval_neutral"


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
        gold_logic_constants=sample.logic_constants,
        gold_logic_predicates=sample.logic_predicates,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert result.format_ok == 1.0
    assert result.correct == 1.0
    assert result.nl_logic_parse == 1.0
    assert result.nl_logic_citation_free_valid == 1.0


def test_nl_exact_trace_translates_to_valid_logic():
    cfg = make_task(template=TemplateName.NL_EXACT)
    builder = TaskBuilder(cfg)
    sample = builder.sample(3, train=False)
    result = OutputEvaluator().evaluate(
        sample.target,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        gold_logic_constants=sample.logic_constants,
        gold_logic_predicates=sample.logic_predicates,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert result.format_ok == 1.0
    assert result.correct == 1.0
    assert result.nl_logic_parse == 1.0
    assert result.nl_logic_citation_free_valid == 1.0


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
