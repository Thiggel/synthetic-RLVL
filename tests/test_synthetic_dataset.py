from __future__ import annotations

import pytest

from logic_engine import LogicEngine
from synthetic_dataset import DatasetConfig, LogicDatasetGenerator


def _extract_conclusion(proof_line: str) -> str:
    return proof_line.split(". ", 1)[1].split(" ; ", 1)[0].strip()


def _extract_formula(line: str) -> str:
    return line.split(". ", 1)[1].split(" ; ", 1)[0].strip()


def _extract_premise_formula(line: str) -> str:
    return line.split(". ", 1)[1].strip()


def _predicate_texts(example) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in example.predicates:
        pred, text = line.split("x:", 1)
        out[pred.strip()] = text.strip().removeprefix("x is ").strip()
    return out


def _formula_text(example, formula: str) -> str:
    return _predicate_texts(example)[formula.strip()[0]]


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


def test_hard_v2_keeps_gold_proof_valid_and_adds_adversarial_premises():
    standard = LogicDatasetGenerator(DatasetConfig(depth=6, distractor_ratio=0.5, seed=123)).generate(0)
    hard = LogicDatasetGenerator(DatasetConfig(depth=6, distractor_ratio=0.5, difficulty="hard_v2", seed=123)).generate(0)

    engine = LogicEngine()
    report = engine.analyze_proof(
        premises="\n".join(hard.premises_fol),
        conclusion=_extract_conclusion(hard.proof_fol[-1]),
        proof="\n".join(hard.proof_fol),
    )

    assert report.ok is True
    assert len(hard.premises_fol) > len(standard.premises_fol)
    assert len(hard.proof_fol) == 1 + hard.metadata["depth"] + hard.metadata["depth"] // 2
    assert hard.metadata["difficulty"] == "hard_v2"
    assert hard.metadata["branching_factor"] == 4
    assert hard.metadata["hard_counts"]["branch_rules"] > 0
    assert hard.metadata["hard_counts"]["wrong_entity_premises"] > 0


def test_hard_v1_is_smaller_than_hard_v2_but_still_adversarial():
    v1 = LogicDatasetGenerator(DatasetConfig(depth=10, distractor_ratio=0.5, difficulty="hard_v1", seed=3407)).generate(0)
    v2 = LogicDatasetGenerator(DatasetConfig(depth=10, distractor_ratio=0.5, difficulty="hard_v2", seed=3407)).generate(0)

    assert len(v1.premises_fol) < len(v2.premises_fol)
    assert v1.metadata["hard_counts"]["branch_rules"] > 0
    assert v1.metadata["hard_counts"]["missing_support_rules"] > 0
    assert v1.metadata["hard_counts"]["wrong_entity_premises"] > 0


@pytest.mark.parametrize("depth", [1, 2, 10, 20])
@pytest.mark.parametrize("difficulty", ["hard_v1", "hard_v2"])
def test_hard_gold_trace_is_clean_and_short(depth: int, difficulty: str):
    hard = LogicDatasetGenerator(DatasetConfig(depth=depth, distractor_ratio=0.5, difficulty=difficulty, seed=3407)).generate(0)
    proof_formulas = [_extract_formula(line) for line in hard.proof_fol]
    premise_formulas = [_extract_premise_formula(line) for line in hard.premises_fol]
    expected_proof_lines = 1 + depth + depth // 2

    assert len(proof_formulas) == expected_proof_lines
    assert len(proof_formulas) == len(set(proof_formulas))
    assert len(premise_formulas) == len(set(premise_formulas))
    assert proof_formulas[-1] not in premise_formulas
    assert hard.metadata["num_distractors"] == 0


@pytest.mark.parametrize("depth", [5, 10, 20])
def test_hard_v3_is_compact_adversarial_and_clean(depth: int):
    hard = LogicDatasetGenerator(DatasetConfig(depth=depth, difficulty="hard_v3", seed=3407)).generate(0)
    proof_formulas = [_extract_formula(line) for line in hard.proof_fol]
    premise_formulas = [_extract_premise_formula(line) for line in hard.premises_fol]
    premise_texts = [line.split(". ", 1)[1].strip() for line in hard.premises_nl]
    counts = hard.metadata["hard_counts"]

    engine = LogicEngine()
    report = engine.analyze_proof(
        premises="\n".join(hard.premises_fol),
        conclusion=_extract_conclusion(hard.proof_fol[-1]),
        proof="\n".join(hard.proof_fol),
    )

    assert report.ok is True
    assert len(proof_formulas) == 1 + depth + depth // 2
    assert len(proof_formulas) == len(set(proof_formulas))
    assert len(premise_formulas) == len(set(premise_formulas))
    assert len(premise_texts) == len(set(premise_texts))
    assert counts["answer_decoys"] > 0
    assert counts["branch_rules"] > 0
    assert counts["missing_support_rules"] > 0
    assert counts["wrong_entity_premises"] > 0
    assert counts["total_adversarial_premises"] <= counts["adversarial_premise_budget"]
    assert hard.metadata["nl_premises_shuffled"] is True


def test_hard_v3_shuffles_only_natural_theory():
    hard = LogicDatasetGenerator(DatasetConfig(depth=6, difficulty="hard_v3", seed=3407)).generate(0)
    premise_formulas = [_extract_premise_formula(line) for line in hard.premises_fol]
    query_const = hard.metadata["query_constant"]

    assert premise_formulas[0].endswith(query_const)
    assert "->" not in premise_formulas[0]
    assert any("For " in line for line in hard.premises_nl[:5])
    assert [line.split(". ", 1)[0] for line in hard.premises_nl] == [
        str(i) for i in range(1, len(hard.premises_nl) + 1)
    ]


@pytest.mark.parametrize("depth", [1, 3, 15, 20])
def test_hard_v5_uses_citation_free_clean_gold_trace(depth: int):
    ex = LogicDatasetGenerator(
        DatasetConfig(depth=depth, difficulty="hard_v5", shortcut_rate=0.8, seed=3407)
    ).generate(0)
    proof = "\n".join(ex.proof_fol)
    conclusion = _extract_conclusion(ex.proof_fol[-1])
    engine = LogicEngine()

    strict = engine.analyze_proof(
        premises="\n".join(ex.premises_fol),
        conclusion=conclusion,
        proof=proof,
    )
    citation_free = engine.analyze_proof_citation_free(
        premises="\n".join(ex.premises_fol),
        conclusion=conclusion,
        proof=proof,
    )

    assert strict.ok is False
    assert citation_free.ok is True
    assert all(",P" not in line and ",L" not in line for line in ex.proof_fol)
    assert len(ex.proof_fol) == 2 * depth + 1
    assert len(ex.premises_fol) <= 4 * depth + 2
    assert ex.metadata["citation_free_gold"] is True
    assert ex.answer == ex.metadata["gold_answer"]
    assert ex.metadata["shortcut_branch_answer"] != ex.metadata["gold_answer"]
    assert ex.metadata["shortcut_path_states"][0] == ex.metadata["path_states"][0]
    assert ex.metadata["shortcut_path_states"][-1] == ex.metadata["shortcut_branch_answer"]

@pytest.mark.parametrize("depth", [1, 3, 10, 20, 25])
def test_hard_fsa_gold_trace_is_citation_free_valid_and_ambiguous(depth: int):
    ex = LogicDatasetGenerator(
        DatasetConfig(depth=depth, difficulty="hard_fsa", branching_factor=4, seed=3407)
    ).generate(0)
    proof = "\n".join(ex.proof_fol)
    conclusion = _extract_conclusion(ex.proof_fol[-1])
    engine = LogicEngine()
    strict = engine.analyze_proof("\n".join(ex.premises_fol), conclusion, proof)
    citation_free = engine.analyze_proof_citation_free("\n".join(ex.premises_fol), conclusion, proof)

    assert strict.ok is False
    assert citation_free.ok is True
    assert len(ex.proof_fol) == 2 * depth + 1
    assert ex.metadata["expected_proof_lines"] == 2 * depth + 1
    assert ex.metadata["final_conclusion_kind"] == "state"
    assert _formula_text(ex, conclusion) == ex.answer
    assert conclusion.endswith(ex.metadata["query_constant"])
    assert len(ex.premises_fol) == 2 + 8 * depth
    assert ex.metadata["shortcut_branch_answer"] != ex.metadata["gold_answer"]
    assert len(ex.metadata["branch_orders"]) == depth
    assert all(len(order) == 4 for order in ex.metadata["branch_orders"])
    branch_states = ex.metadata["branch_states"]
    branch_markers = ex.metadata["branch_markers"]
    path_constants = ex.metadata["path_constants"]
    output_atoms = [
        (branch_states[branch_idx][layer], path_constants[layer])
        for branch_idx in range(4)
        for layer in range(1, depth + 1)
    ]
    assert len(output_atoms) == len(set(output_atoms))
    assert len({states[0] for states in branch_states}) == 1
    assert len({markers[0] for markers in branch_markers}) == 4
    for layer in range(depth):
        antecedents = [
            (branch_markers[branch_idx][layer], branch_states[branch_idx][layer], path_constants[layer])
            for branch_idx in range(4)
        ]
        assert len(antecedents) == len(set(antecedents))
    for layer in range(1, depth + 1):
        assert len({states[layer] for states in branch_states}) == 4
    assert len({states[-1] for states in branch_states}) == 4


@pytest.mark.parametrize("shortcut_rate,enabled", [(1.0, True), (0.0, False)])
def test_hard_fsa_schema_shortcut_mode_and_gold_validity(shortcut_rate: float, enabled: bool):
    ex = LogicDatasetGenerator(
        DatasetConfig(depth=10, difficulty="hard_fsa_schema", branching_factor=4, shortcut_rate=shortcut_rate, seed=3407)
    ).generate(0)
    proof = "\n".join(ex.proof_fol)
    conclusion = _extract_conclusion(ex.proof_fol[-1])
    citation_free = LogicEngine().analyze_proof_citation_free("\n".join(ex.premises_fol), conclusion, proof)

    assert citation_free.ok is True
    assert ex.metadata["final_conclusion_kind"] == "state"
    assert ex.metadata["expected_proof_lines"] == 2 * ex.metadata["depth"] + 1
    assert _formula_text(ex, conclusion) == ex.answer
    assert conclusion.endswith(ex.metadata["query_constant"])
    assert ex.metadata["shortcut_enabled"] is enabled
    assert len(ex.metadata["candidate_answers"]) == 4
    assert len(set(ex.metadata["candidate_answers"])) == 4
    assert ex.metadata["candidate_answers"][ex.metadata["gold_candidate_position"]] == ex.answer
    if enabled:
        assert ex.metadata["schema_prediction_correct"] is True
        assert ex.metadata["schema_predicted_answer"] == ex.answer
        assert "shared_transition_schema" in ex.metadata["shortcut_types"]
    else:
        assert ex.metadata["split_intervention"] == "eval_neutral"


def test_hard_fsa_schema_supports_easy_two_branch_curriculum():
    ex = LogicDatasetGenerator(
        DatasetConfig(depth=5, difficulty="hard_fsa_schema", branching_factor=2, shortcut_rate=1.0, seed=3407)
    ).generate(0)
    proof = "\n".join(ex.proof_fol)
    conclusion = _extract_conclusion(ex.proof_fol[-1])
    citation_free = LogicEngine().analyze_proof_citation_free("\n".join(ex.premises_fol), conclusion, proof)

    assert citation_free.ok is True
    assert ex.metadata["final_conclusion_kind"] == "state"
    assert ex.metadata["expected_proof_lines"] == 2 * ex.metadata["depth"] + 1
    assert _formula_text(ex, conclusion) == ex.answer
    assert conclusion.endswith(ex.metadata["query_constant"])
    assert ex.metadata["shortcut_enabled"] is True
    assert ex.metadata["branching_factor"] == 2
    assert len(ex.metadata["candidate_answers"]) == 2
    assert len(set(ex.metadata["candidate_answers"])) == 2
    assert ex.metadata["candidate_answers"][ex.metadata["gold_candidate_position"]] == ex.answer
    assert all(len(order) == 2 for order in ex.metadata["branch_orders"])
