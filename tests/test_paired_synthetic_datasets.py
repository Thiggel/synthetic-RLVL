from pathlib import Path

import pytest

from synthrlvl.datasets import (
    PAIRED_DATASET_KINDS,
    PairedGeneratorConfig,
    PairedSyntheticGenerator,
    validate_logic_example,
)
from synthrlvl.task import TaskBuilder, task_sample_from_logic_example
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def _task_cfg(template: TemplateName) -> TaskConfig:
    return TaskConfig(
        template=template,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.0,
        train_steps=StepRange(1, 3),
        val_steps=StepRange(1, 3),
        seed=3407,
    )


def test_paired_synthetic_examples_are_logic_valid():
    for kind in PAIRED_DATASET_KINDS:
        if kind == "official_igsm" and not _has_official_igsm_repo():
            pytest.skip("official iGSM repo is not available")
        ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind=kind, depth=3, seed=3407)).generate(0)
        validation = validate_logic_example(ex)
        assert validation.ok, (kind, validation.error, validation.line_errors)
        assert ex.metadata["logic_trace_valid"] is True


def test_official_igsm_uses_official_generator_and_mod23_trace():
    if not _has_official_igsm_repo():
        pytest.skip("official iGSM repo is not available")
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="official_igsm", depth=3, seed=3407)).generate(2)
    validation = validate_logic_example(ex)

    assert validation.ok, (validation.error, validation.line_errors)
    assert ex.metadata["dataset_family"] == "official_igsm"
    assert ex.metadata["official_n_op"] == 3
    assert ex.metadata["modulus"] == 23
    assert "official_problem_text" in ex.metadata
    assert not any("answer_value" in constant for constant in ex.constants)
    assert any("MOD23" in line for line in ex.proof_fol)


def test_official_igsm_targets_use_semantic_bare_variables():
    if not _has_official_igsm_repo():
        pytest.skip("official iGSM repo is not available")
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="official_igsm", depth=5, seed=3407)).generate(0)
    logic_sample = task_sample_from_logic_example(ex, cfg=_task_cfg(TemplateName.LOGIC), depth=5)
    nl_sample = task_sample_from_logic_example(ex, cfg=_task_cfg(TemplateName.NL_EXACT), depth=5)

    assert validate_logic_example(ex).ok
    assert "official iGSM variable" not in logic_sample.target
    assert "v_" not in logic_sample.target
    assert "v_" not in nl_sample.target
    assert "iGSM" not in nl_sample.target
    assert "From the definition of " in nl_sample.target
    assert any(" = the number of each " in constant for constant in ex.constants)


def test_logic_engine_allows_bare_lowercase_registers_only_for_equalities():
    from logic_engine import LogicEngine

    engine = LogicEngine()
    equality_report = engine.analyze_proof(
        "s = 2\nz = 6 + s",
        "z = 8",
        "3. s = 2 ; R,1\n4. z = 6 + s ; R,2\n5. z = 6 + 2 ; =E,3,4\n6. z = 8 ; MOD23,5",
    )
    predicate_report = engine.analyze_proof("P(s)", "P(s)", "2. P(s) ; R,1")

    assert equality_report.ok, equality_report.error
    assert not predicate_report.ok
    assert predicate_report.error is not None


def test_official_igsm_semantic_bare_variable_targets_validate_across_depths():
    if not _has_official_igsm_repo():
        pytest.skip("official iGSM repo is not available")
    from synthrlvl.metrics import OutputEvaluator

    evaluator = OutputEvaluator()
    for depth in (1, 2, 5, 10):
        for index in (0, 1, 2):
            ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="official_igsm", depth=depth, seed=3407)).generate(index)
            validation = validate_logic_example(ex)
            assert validation.ok, (depth, index, validation.error, validation.line_errors)
            for template in (TemplateName.LOGIC, TemplateName.NL_EXACT):
                sample = task_sample_from_logic_example(ex, cfg=_task_cfg(template), depth=depth)
                result = evaluator.evaluate(
                    sample.target,
                    template=template,
                    gold_answer=sample.answer,
                    gold_logic_premises=sample.logic_premises,
                    gold_logic_conclusion=sample.logic_conclusion,
                    gold_logic_constants=sample.logic_constants,
                    gold_logic_predicates=sample.logic_predicates,
                )
                assert result.format_ok == 1.0, (depth, index, template, sample.target)
                assert result.correct == 1.0, (depth, index, template, sample.target)
                if template == TemplateName.LOGIC:
                    assert result.grounded_valid == 1.0, (depth, index, sample.target)
                else:
                    assert result.nl_logic_parse == 1.0, (depth, index, sample.target)
                    assert result.nl_logic_citation_free_valid == 1.0, (depth, index, sample.target)


def test_maze_navigation_is_key_constrained_graph_with_blocked_decoys():
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="maze_navigation", depth=3, seed=3407)).generate(0)
    validation = validate_logic_example(ex)

    assert validation.ok, (validation.error, validation.line_errors)
    assert ex.metadata["dataset_family"] == "maze_navigation"
    assert ex.metadata["task_structure"] == "keyed_constrained_graph"
    assert ex.metadata["requires_key_tracking"] is True
    assert len(ex.metadata["gold_path"]) == 4
    assert len(ex.metadata["key_path"]) == 4
    assert ex.metadata["blocked_edges"]
    assert any("Have0(" in line for line in ex.premises_fol)
    assert any("Finds(" in line for line in ex.premises_fol)
    assert any("Door(" in line and line.count(",") == 2 for line in ex.premises_fol)
    assert ex.answer in ex.metadata["treasure_rooms"]


def test_maze_navigation_uses_typed_room_and_key_symbols():
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="maze_navigation", depth=8, seed=3407)).generate(0)
    validation = validate_logic_example(ex)
    room_symbols = {entry["name"]: entry["symbol"] for entry in ex.metadata["room_symbols"]}
    key_symbols = {entry["name"]: entry["symbol"] for entry in ex.metadata["key_symbols"]}

    assert validation.ok, (validation.error, validation.line_errors)
    assert room_symbols
    assert key_symbols
    assert set(room_symbols.values()).isdisjoint(set(key_symbols.values()))
    assert all(symbol.startswith("r_") for symbol in room_symbols.values())
    assert all(symbol.startswith("k_") for symbol in key_symbols.values())
    assert any(constant.startswith("r_") and " = maze room " in constant for constant in ex.constants)
    assert any(constant.startswith("k_") and " = maze key " in constant for constant in ex.constants)
    assert any("Door(r_" in line and ",k_" in line for line in ex.premises_fol)
    assert any("Finds(r_" in line and ",k_" in line for line in ex.premises_fol)
    assert not any("silver = maze room silver" == constant for constant in ex.constants)
    assert not any("silver = maze key silver" == constant for constant in ex.constants)


def test_attribute_constraints_solve_slots_directly_without_feedback_or_assignments():
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="attribute_constraints", depth=4, seed=3407)).generate(0)
    validation = validate_logic_example(ex)
    metadata = ex.metadata
    slots = metadata["slots"]
    expected_answer = "-".join(slot["value"] for slot in slots)

    assert validation.ok, (validation.error, validation.line_errors)
    assert metadata["dataset_family"] == "attribute_constraints"
    assert metadata["slot_count"] == 4
    assert all("Feedback(" not in line for line in ex.premises_fol)
    assert all("Candidate(" not in line for line in ex.premises_fol)
    assert all("assignment_" not in line for line in ex.constants + ex.premises_fol + ex.proof_fol)
    assert ex.answer == expected_answer
    assert all(f"Value({slot['slot']},{slot['value']})" in ex.proof_fol[-1] for slot in slots)


def test_attribute_constraints_nl_exact_trace_translates_to_valid_logic():
    from synthrlvl.metrics import OutputEvaluator

    evaluator = OutputEvaluator()
    for depth in (4, 10, 25):
        ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="attribute_constraints", depth=depth, seed=3407)).generate(0)
        sample = task_sample_from_logic_example(ex, cfg=_task_cfg(TemplateName.NL_EXACT), depth=depth)
        result = evaluator.evaluate(
            sample.target,
            template=TemplateName.NL_EXACT,
            gold_answer=sample.answer,
            gold_logic_premises=sample.logic_premises,
            gold_logic_conclusion=sample.logic_conclusion,
            gold_logic_constants=sample.logic_constants,
            gold_logic_predicates=sample.logic_predicates,
        )

        assert result.format_ok == 1.0, (depth, sample.target)
        assert result.correct == 1.0, (depth, sample.target)
        assert result.nl_logic_parse == 1.0, (depth, sample.target)
        assert result.nl_logic_citation_free_valid == 1.0, (depth, sample.target)


def test_attribute_constraints_depth_is_not_capped_at_six():
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="attribute_constraints", depth=12, seed=3407)).generate(0)
    validation = validate_logic_example(ex)
    metadata = ex.metadata

    assert validation.ok, (validation.error, validation.line_errors)
    assert metadata["slot_count"] == 8
    assert len(metadata["slots"]) == 8
    assert len(str(ex.answer).split("-")) == 8


def test_constraint_aliases_use_attribute_constraint_generator():
    for kind in ("mastermind_constraints", "constraint_satisfaction", "constraint_propagation"):
        ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind=kind, depth=3, seed=3407)).generate(0)
        validation = validate_logic_example(ex)
        assert validation.ok, (kind, validation.error, validation.line_errors)
        assert ex.metadata["dataset_family"] == "attribute_constraints"


def test_paired_synthetic_examples_use_existing_sft_formatter():
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="graph_traversal", depth=2, seed=3407)).generate(1)
    logic_sample = task_sample_from_logic_example(ex, cfg=_task_cfg(TemplateName.LOGIC), depth=2)
    nl_sample = task_sample_from_logic_example(ex, cfg=_task_cfg(TemplateName.NL_EXACT), depth=2)

    assert "<formal>" in logic_sample.target
    assert "<proof>" in logic_sample.target
    assert "<think>" in nl_sample.target
    assert f"<answer>\n{ex.answer}\n</answer>" in logic_sample.target


def test_official_igsm_logic_sample_with_empty_predicates_is_format_ok():
    if not _has_official_igsm_repo():
        pytest.skip("official iGSM repo is not available")
    ex = PairedSyntheticGenerator(PairedGeneratorConfig(kind="official_igsm", depth=2, seed=17)).generate(0)
    sample = task_sample_from_logic_example(ex, cfg=_task_cfg(TemplateName.LOGIC), depth=2)
    from synthrlvl.metrics import OutputEvaluator

    result = OutputEvaluator().evaluate(
        sample.target,
        template=TemplateName.LOGIC,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        gold_logic_constants=sample.logic_constants,
        gold_logic_predicates=sample.logic_predicates,
    )

    assert "<predicates>\n\n</predicates>" in sample.target
    assert result.format_ok == 1.0
    assert result.grounded_valid == 1.0


def test_task_builder_can_generate_paired_synthetic_difficulties():
    cfg = _task_cfg(TemplateName.LOGIC)
    cfg = TaskConfig(
        template=cfg.template,
        prefill=cfg.prefill,
        distractor_ratio=0.0,
        train_steps=StepRange(2, 2),
        val_steps=StepRange(2, 2),
        seed=3407,
        difficulty="igsm_arithmetic",
    )
    sample = TaskBuilder(cfg).sample(0, train=False)
    assert sample.depth == 2
    assert sample.metadata["dataset_family"] == "igsm_arithmetic"
    assert "<question>" in sample.prompt


def _has_official_igsm_repo() -> bool:
    import os

    candidates = []
    if os.environ.get("IGSM_REPO_PATH"):
        candidates.append(Path(os.environ["IGSM_REPO_PATH"]))
    if os.environ.get("WORK"):
        candidates.append(Path(os.environ["WORK"]) / "codex_research/iGSM")
    return any((path / "data_gen/pretrain/id_gen.py").exists() for path in candidates)
