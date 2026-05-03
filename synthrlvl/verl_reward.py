from __future__ import annotations

from typing import Any

try:
    from .metrics import OutputEvaluator, RewardComputer
    from .types import PrefillMode, RewardSchema, TemplateName
except ImportError:
    # VERL may load this file as a standalone module path in Ray workers.
    from synthrlvl.metrics import OutputEvaluator, RewardComputer
    from synthrlvl.types import PrefillMode, RewardSchema, TemplateName

_EVAL = OutputEvaluator()
_REWARD = RewardComputer(_EVAL)


def _coerce_template(value: str | None) -> TemplateName:
    try:
        return TemplateName(value or TemplateName.LOGIC.value)
    except Exception:
        return TemplateName.LOGIC


def _coerce_prefill(value: str | None) -> PrefillMode:
    try:
        return PrefillMode(value or PrefillMode.NONE.value)
    except Exception:
        return PrefillMode.NONE


def _coerce_schema(value: str | None) -> RewardSchema:
    try:
        return RewardSchema(value or RewardSchema.CORRECT_PLUS_VALID_PLUS_0P1_FORMAT.value)
    except Exception:
        return RewardSchema.CORRECT_PLUS_VALID_PLUS_0P1_FORMAT


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    schema: str | None = None,
    **_: Any,
) -> dict[str, float]:
    del data_source
    info = extra_info or {}

    template = _coerce_template(str(info.get("template", TemplateName.LOGIC.value)))
    prefill = _coerce_prefill(str(info.get("prefill", PrefillMode.NONE.value)))
    reward_schema = _coerce_schema(str(info.get("schema", schema)))

    lines = info.get("gold_first_modality_lines", [])
    if isinstance(lines, str):
        gold_first_modality_lines = [x.strip() for x in lines.splitlines() if x.strip()]
    elif isinstance(lines, list):
        gold_first_modality_lines = [str(x).strip() for x in lines if str(x).strip()]
    else:
        gold_first_modality_lines = []

    gold_logic_premises = str(info.get("gold_logic_premises", ""))
    gold_logic_conclusion = str(info.get("gold_logic_conclusion", ""))

    try:
        score, comp = _REWARD.reward(
            solution_str,
            schema=reward_schema,
            template=template,
            prefill=prefill,
            gold_answer=str(ground_truth),
            gold_logic_premises=gold_logic_premises,
            gold_logic_conclusion=gold_logic_conclusion,
            gold_first_modality_lines=gold_first_modality_lines,
        )
        line_valid = _REWARD._line_valid_fraction(solution_str, template=template)
        return {
            "score": float(score),
            "reward/format": float(comp.format_ok),
            "reward/correct": float(comp.correct),
            "reward/valid": float(comp.valid),
            "reward/line_valid": float(line_valid),
            "reward/line_match": float(comp.line_match),
        }
    except Exception:
        return {
            "score": 0.0,
            "reward/format": 0.0,
            "reward/correct": 0.0,
            "reward/valid": 0.0,
            "reward/line_match": 0.0,
        }
