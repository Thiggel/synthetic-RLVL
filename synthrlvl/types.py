from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TemplateName(str, Enum):
    LOGIC = "logic"
    NATURAL = "natural"
    LOGIC_NATURAL = "logic_natural"
    NATURAL_LOGIC = "natural_logic"
    NL_EXACT = "nl_exact"
    FORMAL_THINK = "formal_think"
    THINK_FORMAL = "think_formal"


class PrefillMode(str, Enum):
    NONE = "none"
    GOLD = "gold"
    LINE_REWARD = "line_reward"


class RewardSchema(str, Enum):
    CORRECT_PLUS_0P1_FORMAT = "correct_plus_0p1_format"
    INDICATOR_CORRECT_AND_FORMAT = "indicator_correct_and_format"
    CORRECT_PLUS_VALID_PLUS_0P1_FORMAT = "correct_plus_valid_plus_0p1_format"
    CORRECT_PLUS_LINE_VALID_PLUS_0P1_FORMAT = "correct_plus_line_valid_plus_0p1_format"
    CORRECT_PLUS_0P75_VALID_PLUS_0P1_FORMAT = "correct_plus_0p75_valid_plus_0p1_format"
    CORRECT_PLUS_0P5_VALID_PLUS_0P1_FORMAT = "correct_plus_0p5_valid_plus_0p1_format"
    CORRECT_PLUS_0P25_VALID_PLUS_0P1_FORMAT = "correct_plus_0p25_valid_plus_0p1_format"
    INDICATOR_ALL = "indicator_all"


@dataclass(frozen=True)
class StepRange:
    min_step: int
    max_step: int


@dataclass(frozen=True)
class TaskConfig:
    template: TemplateName
    prefill: PrefillMode
    distractor_ratio: float
    train_steps: StepRange
    val_steps: StepRange
    seed: int
