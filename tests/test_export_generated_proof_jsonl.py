from scripts.data.export_generated_proof_jsonl import format_neutral_solution_document
from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def _sample(template: TemplateName):
    config = TaskConfig(
        template=template,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.0,
        train_steps=StepRange(1, 3),
        val_steps=StepRange(1, 3),
        seed=3407,
        difficulty="hard_fsa_schema",
        branching_factor=4,
    )
    return TaskBuilder(config).sample(0, train=True)


def test_neutral_solution_outer_format_matches_across_modalities() -> None:
    logic = format_neutral_solution_document(_sample(TemplateName.LOGIC), TemplateName.LOGIC)
    nl = format_neutral_solution_document(_sample(TemplateName.NL_EXACT), TemplateName.NL_EXACT)

    for text in (logic, nl):
        assert "\n\nSolution:\nContext:\n" in text
        assert "\n\nDerivation:\n" in text
        assert "\n\nConclusion:\n" in text
        assert "\n\nFinal answer: " in text
        assert "<formal>" not in text
        assert "<think>" not in text
        assert "<question>" not in text
        assert "<answer>" not in text

    assert logic.split("\n\nSolution:\n", 1)[0] == nl.split("\n\nSolution:\n", 1)[0]
    assert "Constants:\n" in logic
    assert "Predicates:\n" in logic
    assert "Premises:\n" in logic
    assert "Constants:\n" not in nl
    assert "Predicates:\n" not in nl
    assert "Premises:\n" in nl


def test_neutral_solution_preserves_answer_and_conclusion() -> None:
    sample = _sample(TemplateName.LOGIC)
    text = format_neutral_solution_document(sample, TemplateName.LOGIC)
    assert f"Final answer: {sample.answer}" in text
    assert sample.metadata["query_constant"] in text
