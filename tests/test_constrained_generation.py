from __future__ import annotations

from collections.abc import Sequence

from synthrlvl.generation import ConstrainedProofGenerationConfig, ConstrainedProofGenerator


def test_constrained_generator_selects_valid_novel_line():
    calls: list[tuple[Sequence[str], int, int, float]] = []

    def fake_generate_many(prompts: Sequence[str], max_new_tokens: int, num_samples: int, temperature: float) -> list[list[str]]:
        calls.append((prompts, max_new_tokens, num_samples, temperature))
        if len(calls) == 1:
            return [[
                "<formal>\n"
                "<constants>\na = Alice\n</constants>\n"
                "<predicates>\nPa: Alice is P\nQa: Alice is Q\n</predicates>\n"
                "<premises>\nP(a)->Q(a)\nP(a)\n</premises>\n"
                "<proof>\n"
            ]]
        if num_samples > 1:
            return [["Q(a) ; IE, 1,3", "Q(a) ; IE, 1,2"]]
        return [["<conclusion>\nQ(a)\n</conclusion>\n</formal>\n<answer>\nQ\n</answer>"]]

    generator = ConstrainedProofGenerator(generate_many=fake_generate_many)
    outputs, traces = generator.generate_many(
        ["<question>dummy</question>\n"],
        max_new_tokens=128,
        config=ConstrainedProofGenerationConfig(
            num_generations=1,
            candidates_per_line=2,
            max_lines=1,
            max_line_tokens=32,
            suffix_max_tokens=64,
            temperature=1.0,
        ),
    )

    assert "Q(a) ; IE, 1,2" in outputs[0][0]
    assert "<question>" not in outputs[0][0]
    assert traces[0][0].best_scores == (3,)
