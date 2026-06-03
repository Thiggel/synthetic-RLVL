from __future__ import annotations

import pytest

from synthetic_dataset import DatasetConfig, LogicDatasetGenerator
from synthrlvl.metrics import OutputEvaluator, RewardComputer
from synthrlvl.sft_data import build_sft_dataset_from_materialized_rows
from synthrlvl.task import TaskBuilder, task_sample_from_logic_example
from synthrlvl.types import PrefillMode, RewardSchema, StepRange, TaskConfig, TemplateName
from train_sft import BalancedModalityAccumulationSampler


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


@pytest.mark.parametrize(
    ("template", "block_tag", "marker"),
    [
        (TemplateName.TERSE_NL, "think", None),
        (TemplateName.RULE_ANNOTATED_NL, "think", "[rule:"),
        (TemplateName.PSEUDOCODE, "think", "step_1: derive"),
        (TemplateName.SHUFFLED_NL, "think", None),
        (TemplateName.SHUFFLED_LOGIC, "formal", None),
        (TemplateName.INVALID_LOGIC, "formal", " ; R"),
    ],
)
def test_ablation_templates_emit_scorable_targets(template: TemplateName, block_tag: str, marker: str | None):
    gen = LogicDatasetGenerator(DatasetConfig(depth=3, distractor_ratio=0.5, seed=123))
    ex = gen.generate(0)
    cfg = make_task(template=template)
    sample = task_sample_from_logic_example(ex, cfg=cfg, depth=3)

    assert f"<{block_tag}>" in sample.target
    assert "<answer>" in sample.target
    if marker is not None:
        assert marker in sample.target

    result = OutputEvaluator().evaluate(
        sample.target,
        template=template,
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


@pytest.mark.parametrize("template", [TemplateName.RULE_ANNOTATED_NL, TemplateName.PSEUDOCODE])
def test_controlled_nl_trace_wrappers_translate_to_valid_logic(template: TemplateName):
    gen = LogicDatasetGenerator(DatasetConfig(depth=3, distractor_ratio=0.5, seed=123))
    ex = gen.generate(0)
    cfg = make_task(template=template)
    sample = task_sample_from_logic_example(ex, cfg=cfg, depth=3)

    result = OutputEvaluator().evaluate(
        sample.target,
        template=template,
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


def test_invalid_logic_template_breaks_proof_validity():
    gen = LogicDatasetGenerator(DatasetConfig(depth=3, distractor_ratio=0.5, seed=123))
    ex = gen.generate(0)
    cfg = make_task(template=TemplateName.INVALID_LOGIC)
    sample = task_sample_from_logic_example(ex, cfg=cfg, depth=3)

    result = OutputEvaluator().evaluate(
        sample.target,
        template=cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    assert result.valid == 0.0


def test_high_depth_shortcut_schema_extended_predicates_are_scorable():
    gen = LogicDatasetGenerator(
        DatasetConfig(
            depth=25,
            difficulty="hard_fsa_schema",
            branching_factor=4,
            shortcut_rate=1.0,
            seed=3407,
        )
    )
    ex = gen.generate(0)
    assert any("(x):" in line for line in ex.predicates)

    cfg = make_task(template=TemplateName.NL_EXACT)
    sample = task_sample_from_logic_example(ex, cfg=cfg, depth=25)
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


@pytest.mark.parametrize(
    ("template", "mode", "block_tag"),
    [
        (TemplateName.CONDITIONED_LOGIC, "formal_logic", "formal"),
        (TemplateName.CONDITIONED_NL, "natural_language", "think"),
    ],
)
def test_conditioned_templates_add_mode_prompt(template: TemplateName, mode: str, block_tag: str):
    gen = LogicDatasetGenerator(DatasetConfig(depth=3, distractor_ratio=0.5, seed=123))
    ex = gen.generate(0)
    cfg = make_task(template=template)
    sample = task_sample_from_logic_example(ex, cfg=cfg, depth=3)

    assert sample.prompt.startswith(f"<reasoning_mode>\n{mode}\n</reasoning_mode>\n")
    assert f"<{block_tag}>" in sample.target
    assert "<answer>" in sample.target


def test_conditioned_dual_materialized_training_duplicates_modalities():
    class TinyTokenizer:
        eos_token_id = 0

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            return {"input_ids": [ord(ch) % 251 + 1 for ch in text]}

    gen = LogicDatasetGenerator(DatasetConfig(depth=2, distractor_ratio=0.5, seed=123))
    row = gen.generate(0).to_dict()
    row["depth"] = 2
    row["record_index"] = 0
    cfg = make_task(template=TemplateName.CONDITIONED_DUAL)
    bundle = build_sft_dataset_from_materialized_rows(
        train_rows=[row],
        eval_rows=[row],
        task_cfg=cfg,
        tokenizer=TinyTokenizer(),
        max_length=8192,
    )

    assert len(bundle.train) == 2
    assert len(bundle.eval) == 2
    assert bundle.train[0]["_sft_modality"] == "logic"
    assert bundle.train[1]["_sft_modality"] == "nl"


def test_balanced_modality_accumulation_sampler_balances_each_window():
    dataset = {
        "_sft_modality": ["logic"] * 8 + ["nl"] * 8,
    }
    sampler = BalancedModalityAccumulationSampler(dataset, accumulation_steps=4, seed=3407)
    indices = list(sampler)

    assert len(indices) == 16
    for offset in range(0, len(indices), 4):
        modalities = [dataset["_sft_modality"][index] for index in indices[offset : offset + 4]]
        assert modalities.count("logic") == 2
        assert modalities.count("nl") == 2


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
    assert result.grounded_valid == 1.0
    assert result.citation_free_grounded_valid == 1.0


def test_output_evaluator_distinguishes_internal_from_grounded_validity():
    output = """
<formal>
<constants>
a = entity a
</constants>
<predicates>
P(x): x is P
Q(x): x is Q
R(x): x is R
</predicates>
<premises>
P(a)
P(a) -> Q(a)
</premises>
<proof>
P(a) ; R,1
Q(a) ; ->E,2,3
</proof>
<conclusion>
Q(a)
</conclusion>
</formal>
<answer>
target
</answer>
"""
    result = OutputEvaluator().evaluate(
        output,
        template=TemplateName.LOGIC,
        gold_answer="target",
        gold_logic_premises="P(a)\nP(a) -> R(a)",
        gold_logic_conclusion="R(a)",
        prefill=PrefillMode.NONE,
        gold_first_modality_lines=[],
    )

    assert result.valid == 1.0
    assert result.citation_free_valid == 1.0
    assert result.grounded_valid == 0.0
    assert result.citation_free_grounded_valid == 0.0


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


def test_igsm_nl_exact_trace_translates_to_valid_logic():
    output = """<think>
<premises>
The number of each Grizzly Bear's Pericardium equals 20.
The number of each African Elephant's Pericardium equals 9 times as much as each Grizzly Bear's Pericardium.
</premises>
<proof>
From the definition of Grizzly Bear's Pericardium (k), k equals 20.
From the definition of African Elephant's Pericardium (V), V equals 9 * k.
Substitute k = 20 into the current expression.
Evaluate the arithmetic modulo 23 to get V = 19.
</proof>
<conclusion>
Evaluate the arithmetic modulo 23 to get V = 19.
</conclusion>
</think>
<answer>
19
</answer>"""
    result = OutputEvaluator().evaluate(
        output,
        template=TemplateName.NL_EXACT,
        gold_answer="19",
        gold_logic_premises="k = 20\nV = 9 * k",
        gold_logic_conclusion="V = 19",
        gold_logic_constants=(
            "k = the number of each Grizzly Bear's Pericardium\n"
            "V = the number of each African Elephant's Pericardium"
        ),
        gold_logic_predicates="",
    )

    assert result.format_ok == 1.0
    assert result.correct == 1.0
    assert result.nl_logic_parse == 1.0
    assert result.nl_logic_citation_free_valid == 1.0


def test_igsm_legacy_v_prefixed_trace_still_translates_to_valid_logic():
    output = """<think>
<premises>
The number of each Grizzly Bear's Pericardium equals 20.
The number of each African Elephant's Pericardium equals 9 times as much as each Grizzly Bear's Pericardium.
</premises>
<proof>
From the official iGSM relation, v_k equals 20.
From the iGSM definition of African Elephant's Pericardium (v_v), v_v equals 9 * v_k.
Substitute k = 20 into the current expression.
Evaluate the arithmetic modulo 23 to get v_v = 19.
</proof>
<conclusion>
Evaluate the arithmetic modulo 23 to get v_v = 19.
</conclusion>
</think>
<answer>
19
</answer>"""
    result = OutputEvaluator().evaluate(
        output,
        template=TemplateName.NL_EXACT,
        gold_answer="19",
        gold_logic_premises="v_k = 20\nv_v = 9 * v_k",
        gold_logic_conclusion="v_v = 19",
        gold_logic_constants=(
            "v_k = the number of each Grizzly Bear's Pericardium\n"
            "v_v = the number of each African Elephant's Pericardium"
        ),
        gold_logic_predicates="",
    )

    assert result.format_ok == 1.0
    assert result.correct == 1.0
    assert result.nl_logic_parse == 1.0
    assert result.nl_logic_citation_free_valid == 1.0


def test_igsm_nl_exact_trace_must_match_gold_chain():
    output = """<think>
<premises>
The number of each Grizzly Bear's Pericardium equals 20.
</premises>
<proof>
From the definition of unrelated quantity (m), m equals 20.
Evaluate the arithmetic modulo 23 to get m = 20.
</proof>
<conclusion>
Evaluate the arithmetic modulo 23 to get m = 20.
</conclusion>
</think>
<answer>
20
</answer>"""
    result = OutputEvaluator().evaluate(
        output,
        template=TemplateName.NL_EXACT,
        gold_answer="20",
        gold_logic_premises="k = 20",
        gold_logic_conclusion="k = 20",
        gold_logic_constants=(
            "k = the number of each Grizzly Bear's Pericardium\n"
            "m = the number of each unrelated quantity"
        ),
        gold_logic_predicates="",
    )

    assert result.format_ok == 1.0
    assert result.correct == 1.0
    assert result.nl_logic_parse == 1.0
    assert result.nl_logic_citation_free_valid == 0.0


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
