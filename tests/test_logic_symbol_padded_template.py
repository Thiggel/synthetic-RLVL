from synthrlvl.metrics import OutputEvaluator
from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def test_logic_symbol_padded_template_keeps_trace_valid():
    cfg = TaskConfig(
        template=TemplateName.LOGIC_SYMBOL_PADDED,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(3, 3),
        val_steps=StepRange(3, 3),
        seed=3407,
        difficulty="hard_fsa_schema",
        branching_factor=4,
        require_unique_solution=True,
    )
    sample = TaskBuilder(cfg).sample(0, train=False)
    result = OutputEvaluator().evaluate(
        sample.target,
        template=TemplateName.LOGIC_SYMBOL_PADDED,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        gold_logic_constants=sample.logic_constants,
        gold_logic_predicates=sample.logic_predicates,
    )

    assert "P" in sample.logic_premises
    assert "(c" in sample.logic_premises
    assert result.format_ok == 1.0
    assert result.citation_free_grounded_valid == 1.0


def test_logic_wordified_template_keeps_trace_valid():
    cfg = TaskConfig(
        template=TemplateName.LOGIC_WORDIFIED,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.5,
        train_steps=StepRange(3, 3),
        val_steps=StepRange(3, 3),
        seed=3407,
        difficulty="hard_fsa_schema",
        branching_factor=4,
        require_unique_solution=True,
    )
    sample = TaskBuilder(cfg).sample(0, train=False)
    result = OutputEvaluator().evaluate(
        sample.target,
        template=TemplateName.LOGIC_WORDIFIED,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        gold_logic_constants=sample.logic_constants,
        gold_logic_predicates=sample.logic_predicates,
    )

    assert "(" in sample.logic_premises
    assert "object_" not in sample.logic_premises
    assert result.format_ok == 1.0
    assert result.citation_free_grounded_valid == 1.0
