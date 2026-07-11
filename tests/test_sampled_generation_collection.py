from __future__ import annotations

from types import MethodType, SimpleNamespace

from synthrlvl.eval_loop import UnifiedEvaluator, _EvalPromptRecord
from synthrlvl.types import PrefillMode, TemplateName


def _record(step: int, prompt: str) -> _EvalPromptRecord:
    return _EvalPromptRecord(
        step=step,
        prompt=prompt,
        gold_answer="yes",
        template=TemplateName.LOGIC,
        prefill=PrefillMode.NONE,
        gold_logic_constants="constant a",
        gold_logic_predicates="predicate A(_)",
        gold_logic_premises="A(a)",
        gold_logic_conclusion="A(a)",
        gold_first_modality_lines=[],
        metadata={},
    )


def test_sampled_collection_covers_prompts_before_later_sample_indices():
    evaluator = UnifiedEvaluator()

    def build_sample(self, *, rec, gen, score, source):
        return {
            "step": rec.step,
            "prompt": rec.prompt,
            "generation": gen,
            "source": source,
        }

    evaluator._build_synthetic_sample = MethodType(build_sample, evaluator)
    records = [_record(1, "p1a"), _record(1, "p1b"), _record(2, "p2a")]
    generations = [["1a-0", "1a-1"], ["1b-0", "1b-1"], ["2a-0", "2a-1"]]
    eval_cfg = SimpleNamespace(step_values=lambda: [1, 2])

    samples = evaluator._collect_sampled_generation_examples(
        records=records,
        generations_by_record=generations,
        eval_cfg=eval_cfg,
        collect_samples=6,
    )

    assert [(sample["generation"], sample["sample_index"]) for sample in samples] == [
        ("1a-0", 0),
        ("2a-0", 0),
        ("1b-0", 0),
        ("2a-1", 1),
        ("1a-1", 1),
        ("1b-1", 1),
    ]
    assert all(sample["source"] == "synthetic_sampled" for sample in samples)
