from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from logic_engine import IncrementalProofValidator

from .proof_segments import compose_final_output, draft_from_generation


GenerateMany = Callable[[Sequence[str], int, int, float], list[list[str]]]


@dataclass(frozen=True)
class ConstrainedProofGenerationConfig:
    num_generations: int
    candidates_per_line: int
    max_lines: int
    max_line_tokens: int
    suffix_max_tokens: int
    temperature: float


@dataclass(frozen=True)
class ConstrainedGenerationTrace:
    used_constrained_proof: bool
    proof_lines: int
    candidate_calls: int
    best_scores: tuple[int, ...]


class ConstrainedProofGenerator:
    """Line-level proof rejection/ranking generator.

    The language model proposes a fixed number of candidate next proof lines.
    The logic engine ranks candidates as:
    random < syntactic < valid < valid-and-novel.
    """

    def __init__(self, *, generate_many: GenerateMany, validator: IncrementalProofValidator | None = None):
        self._generate_many = generate_many
        self._validator = validator or IncrementalProofValidator()

    def generate_many(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        config: ConstrainedProofGenerationConfig,
    ) -> tuple[list[list[str]], list[list[ConstrainedGenerationTrace]]]:
        grouped: list[list[str]] = [[] for _ in prompts]
        traces: list[list[ConstrainedGenerationTrace]] = [[] for _ in prompts]
        total = max(1, int(config.num_generations))

        for sample_idx in range(total):
            drafts = self._generate_many(
                prompts,
                max_new_tokens,
                1,
                config.temperature,
            )
            flat_drafts = [items[0] if items else "" for items in drafts]
            for prompt_idx, (prompt, draft) in enumerate(zip(prompts, flat_drafts, strict=True)):
                final, trace = self._constrain_one(
                    prompt=prompt,
                    draft=draft,
                    config=config,
                )
                grouped[prompt_idx].append(final)
                traces[prompt_idx].append(trace)
            print(
                f"[syntheval] constrained sample {sample_idx + 1}/{total} "
                f"({len(prompts)} prompts, candidates_per_line={config.candidates_per_line})",
                flush=True,
            )
        return grouped, traces

    def _constrain_one(
        self,
        *,
        prompt: str,
        draft: str,
        config: ConstrainedProofGenerationConfig,
    ) -> tuple[str, ConstrainedGenerationTrace]:
        parsed = draft_from_generation(prompt, draft)
        if not parsed.available:
            return draft, ConstrainedGenerationTrace(False, 0, 0, ())

        state = self._validator.initial_state(parsed.premises)
        accepted: list[str] = []
        best_scores: list[int] = []
        candidate_calls = 0

        for _ in range(max(1, int(config.max_lines))):
            prefix = parsed.prompt_prefix_through_proof + "\n".join(accepted)
            if accepted:
                prefix += "\n"
            candidates = self._generate_many(
                [prefix],
                config.max_line_tokens,
                max(1, int(config.candidates_per_line)),
                config.temperature,
            )[0]
            candidate_calls += 1
            reports = [self._validator.check_next_line(state, cand) for cand in candidates]
            best = max(reports, key=lambda item: item.score, default=None)
            if best is None or not best.line:
                break
            best_scores.append(best.score)
            if best.line.lower().startswith("</proof>"):
                break
            # Keep the best candidate even if it is only syntactic/random: this
            # preserves a fixed-N scientific intervention rather than looping
            # until success.
            accepted.append(best.line)
            if best.valid:
                state = self._validator.accept_line(state, best.line)
            if len(accepted) >= config.max_lines:
                break

        suffix_prompt = parsed.prompt_prefix_through_proof
        if accepted:
            suffix_prompt += "\n".join(accepted) + "\n"
        suffix_prompt += "</proof>\n"
        suffix = self._generate_many(
            [suffix_prompt],
            config.suffix_max_tokens,
            1,
            config.temperature,
        )[0][0]
        final = compose_final_output(prefix_through_proof=parsed.generation_prefix_through_proof, proof_lines=accepted, suffix=suffix)
        return final, ConstrainedGenerationTrace(
            used_constrained_proof=True,
            proof_lines=len(accepted),
            candidate_calls=candidate_calls,
            best_scores=tuple(best_scores),
        )
