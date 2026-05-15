from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import comb

from synthrlvl.metrics import EvalResult, OutputEvaluator


@dataclass(frozen=True)
class ScoredSample:
    syntactic: float
    format_ok: float
    correct: float
    valid: float
    citation_free_valid: float
    nl_logic_parse: float
    nl_logic_citation_free_valid: float
    joint: float
    citation_free_joint: float
    nl_logic_joint: float

    @classmethod
    def from_eval_result(cls, result: EvalResult) -> "ScoredSample":
        syntactic = float(result.syntactic > 0)
        format_ok = float(result.format_ok > 0)
        correct = float(result.correct > 0)
        valid = float(result.valid > 0)
        citation_free_valid = float(result.citation_free_valid > 0)
        nl_logic_parse = float(result.nl_logic_parse >= 1.0)
        nl_logic_citation_free_valid = float(result.nl_logic_citation_free_valid > 0)
        return cls(
            syntactic=syntactic,
            format_ok=format_ok,
            correct=correct,
            valid=valid,
            citation_free_valid=citation_free_valid,
            nl_logic_parse=nl_logic_parse,
            nl_logic_citation_free_valid=nl_logic_citation_free_valid,
            joint=float(format_ok > 0 and correct > 0 and valid > 0),
            citation_free_joint=float(format_ok > 0 and correct > 0 and citation_free_valid > 0),
            nl_logic_joint=float(format_ok > 0 and correct > 0 and nl_logic_citation_free_valid > 0),
        )


@dataclass(frozen=True)
class PromptSampleScores:
    step: int
    samples: tuple[ScoredSample, ...]


def pass_at_k_estimate(num_samples: int, num_successes: int, k: int) -> float:
    """Unbiased pass@k estimator used by HumanEval when num_samples >= k."""
    n = int(num_samples)
    c = int(num_successes)
    kk = int(k)
    if kk < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < kk:
        raise ValueError(f"pass@{kk} requires at least {kk} samples, got {n}")
    if c <= 0:
        return 0.0
    if n - c < kk:
        return 1.0
    return 1.0 - (comb(n - c, kk) / comb(n, kk))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _score_prompt_samples(
    *,
    records: Sequence[object],
    generations_by_record: Sequence[Sequence[str]],
    output_eval: OutputEvaluator,
) -> list[PromptSampleScores]:
    scores: list[PromptSampleScores] = []
    for rec, generations in zip(records, generations_by_record, strict=True):
        sample_scores: list[ScoredSample] = []
        for gen in generations:
            result = output_eval.evaluate(
                gen,
                template=rec.template,
                gold_answer=rec.gold_answer,
                gold_logic_premises=rec.gold_logic_premises,
                gold_logic_conclusion=rec.gold_logic_conclusion,
                gold_logic_constants=getattr(rec, "gold_logic_constants", ""),
                gold_logic_predicates=getattr(rec, "gold_logic_predicates", ""),
                prefill=rec.prefill,
                gold_first_modality_lines=rec.gold_first_modality_lines,
            )
            sample_scores.append(ScoredSample.from_eval_result(result))
        scores.append(PromptSampleScores(step=int(rec.step), samples=tuple(sample_scores)))
    return scores


def _metric_counts(samples: Sequence[ScoredSample]) -> dict[str, int]:
    return {
        "format": sum(1 for s in samples if s.format_ok > 0),
        "syntactic": sum(1 for s in samples if s.syntactic > 0),
        "correct": sum(1 for s in samples if s.correct > 0),
        "valid": sum(1 for s in samples if s.valid > 0),
        "citation_free_valid": sum(1 for s in samples if s.citation_free_valid > 0),
        "nl_logic_parse": sum(1 for s in samples if s.nl_logic_parse > 0),
        "nl_logic_citation_free_valid": sum(1 for s in samples if s.nl_logic_citation_free_valid > 0),
        "joint": sum(1 for s in samples if s.joint > 0),
        "citation_free_joint": sum(1 for s in samples if s.citation_free_joint > 0),
        "nl_logic_joint": sum(1 for s in samples if s.nl_logic_joint > 0),
    }


def _add_passk_metrics_for_group(
    *,
    metrics: dict[str, float],
    prefix: str,
    group_scores: Sequence[PromptSampleScores],
    k_values: Sequence[int],
) -> None:
    by_k: dict[int, dict[str, list[float]]] = {
        int(k): {
            "format": [],
            "syntactic": [],
            "correct": [],
            "valid": [],
            "citation_free_valid": [],
            "nl_logic_parse": [],
            "nl_logic_citation_free_valid": [],
            "joint": [],
            "citation_free_joint": [],
            "nl_logic_joint": [],
        }
        for k in k_values
    }

    for prompt_scores in group_scores:
        counts = _metric_counts(prompt_scores.samples)
        n = len(prompt_scores.samples)
        for k in k_values:
            kk = int(k)
            if n < kk:
                continue
            for name, count in counts.items():
                by_k[kk][name].append(pass_at_k_estimate(n, count, kk))

    for k in k_values:
        kk = int(k)
        correct = _mean(by_k[kk]["correct"])
        joint = _mean(by_k[kk]["joint"])
        citation_free_joint = _mean(by_k[kk]["citation_free_joint"])
        nl_logic_joint = _mean(by_k[kk]["nl_logic_joint"])
        valid = _mean(by_k[kk]["valid"])
        citation_free_valid = _mean(by_k[kk]["citation_free_valid"])
        nl_logic_parse = _mean(by_k[kk]["nl_logic_parse"])
        nl_logic_citation_free_valid = _mean(by_k[kk]["nl_logic_citation_free_valid"])
        fmt = _mean(by_k[kk]["format"])
        syntactic = _mean(by_k[kk]["syntactic"])
        metrics[f"{prefix}/format_pass@{kk}"] = fmt
        metrics[f"{prefix}/syntactic_pass@{kk}"] = syntactic
        metrics[f"{prefix}/correct_pass@{kk}"] = correct
        metrics[f"{prefix}/valid_pass@{kk}"] = valid
        metrics[f"{prefix}/citation_free_valid_pass@{kk}"] = citation_free_valid
        metrics[f"{prefix}/nl_logic_parse_pass@{kk}"] = nl_logic_parse
        metrics[f"{prefix}/nl_logic_citation_free_valid_pass@{kk}"] = nl_logic_citation_free_valid
        metrics[f"{prefix}/joint_pass@{kk}"] = joint
        metrics[f"{prefix}/citation_free_joint_pass@{kk}"] = citation_free_joint
        metrics[f"{prefix}/nl_logic_joint_pass@{kk}"] = nl_logic_joint
        metrics[f"{prefix}/valid_given_correct@{kk}"] = joint / correct if correct > 0 else 0.0
        metrics[f"{prefix}/correct_given_valid@{kk}"] = joint / valid if valid > 0 else 0.0
        metrics[f"{prefix}/invalid_but_correct@{kk}"] = max(0.0, correct - joint)
        metrics[f"{prefix}/valid_but_wrong@{kk}"] = max(0.0, valid - joint)
        metrics[f"{prefix}/citation_free_valid_given_correct@{kk}"] = (
            citation_free_joint / correct if correct > 0 else 0.0
        )
        metrics[f"{prefix}/nl_logic_valid_given_correct@{kk}"] = (
            nl_logic_joint / correct if correct > 0 else 0.0
        )
        metrics[f"{prefix}/correct_given_citation_free_valid@{kk}"] = (
            citation_free_joint / citation_free_valid if citation_free_valid > 0 else 0.0
        )
        metrics[f"{prefix}/correct_given_nl_logic_valid@{kk}"] = (
            nl_logic_joint / nl_logic_citation_free_valid if nl_logic_citation_free_valid > 0 else 0.0
        )
        metrics[f"{prefix}/citation_free_invalid_but_correct@{kk}"] = max(0.0, correct - citation_free_joint)
        metrics[f"{prefix}/citation_free_valid_but_wrong@{kk}"] = max(0.0, citation_free_valid - citation_free_joint)


def score_pass_at_k(
    *,
    records: Sequence[object],
    generations_by_record: Sequence[Sequence[str]],
    output_eval: OutputEvaluator,
    k_values: Sequence[int],
    metric_prefix: str = "synthetic_sampled",
    band_predicates: dict[str, Callable[[int], bool]] | None = None,
) -> dict[str, float]:
    """Score sampled generations by depth and optional OOD bands."""
    if len(records) != len(generations_by_record):
        raise RuntimeError(
            f"Mismatched sampled eval lengths: {len(records)} records vs "
            f"{len(generations_by_record)} generation groups"
        )
    sorted_k = sorted({int(k) for k in k_values})
    if not sorted_k:
        return {}

    prompt_scores = _score_prompt_samples(
        records=records,
        generations_by_record=generations_by_record,
        output_eval=output_eval,
    )

    metrics: dict[str, float] = {}
    by_step: dict[int, list[PromptSampleScores]] = defaultdict(list)
    for item in prompt_scores:
        by_step[item.step].append(item)

    for step in sorted(by_step):
        _add_passk_metrics_for_group(
            metrics=metrics,
            prefix=f"{metric_prefix}/step_{step}",
            group_scores=by_step[step],
            k_values=sorted_k,
        )

    if band_predicates:
        for band_name, predicate in band_predicates.items():
            group = [item for item in prompt_scores if predicate(item.step)]
            if group:
                _add_passk_metrics_for_group(
                    metrics=metrics,
                    prefix=f"{metric_prefix}/band_{band_name}",
                    group_scores=group,
                    k_values=sorted_k,
                )

    return metrics


def chunked(iterable: Sequence[object], size: int) -> Iterable[Sequence[object]]:
    step = max(1, int(size))
    for i in range(0, len(iterable), step):
        yield iterable[i : i + step]
