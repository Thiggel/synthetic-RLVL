from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from synthrlvl.datasets import DatasetConfig, LogicDatasetGenerator, LogicExample

from .types import PrefillMode, TaskConfig, TemplateName


@dataclass(frozen=True)
class TaskSample:
    prompt: str
    target: str
    depth: int
    answer: str
    logic_constants: str
    logic_predicates: str
    logic_premises: str
    logic_conclusion: str
    gold_first_modality_lines: List[str]
    metadata: Dict[str, Any]


def _seeded_rng(seed: int, index: int) -> random.Random:
    h = hashlib.sha256(f"{seed}|{index}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _join_unnumbered(lines: List[str]) -> str:
    out = []
    for line in lines:
        stripped = line.strip()
        if ". " in stripped:
            out.append(stripped.split(". ", 1)[1])
        else:
            out.append(stripped)
    return "\n".join(out)


def _extract_facts_rules(premises_nl: List[str]) -> tuple[List[str], List[str]]:
    facts: List[str] = []
    rules: List[str] = []
    for raw in premises_nl:
        text = raw.split(". ", 1)[1].strip() if ". " in raw else raw.strip()
        if text.startswith("All things") or text.startswith("For "):
            rules.append(text)
        else:
            facts.append(text)
    return facts, rules


class TaskBuilder:
    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        self._gens: Dict[tuple[int, bool, float], LogicDatasetGenerator] = {}

    def _generator(self, depth: int, *, train: bool) -> LogicDatasetGenerator:
        shortcut_rate = self.cfg.shortcut_rate
        if self.cfg.difficulty == "hard_fsa_schema" and not train:
            # Eval is intentionally shortcut-neutral; train may be shortcut-rich.
            shortcut_rate = 0.0
        key = (depth, train, shortcut_rate)
        if key not in self._gens:
            ds_cfg = DatasetConfig(
                depth=depth,
                distractor_ratio=self.cfg.distractor_ratio,
                difficulty=self.cfg.difficulty,
                branching_factor=self.cfg.branching_factor,
                decoy_chains=self.cfg.decoy_chains,
                near_miss_ratio=self.cfg.near_miss_ratio,
                side_chain_depth=self.cfg.side_chain_depth,
                entity_decoy_ratio=self.cfg.entity_decoy_ratio,
                answer_decoy_ratio=self.cfg.answer_decoy_ratio,
                shortcut_rate=shortcut_rate,
                require_unique_solution=self.cfg.require_unique_solution,
                seed=self.cfg.seed,
            )
            self._gens[key] = LogicDatasetGenerator(ds_cfg)
        return self._gens[key]

    def _choose_depth(self, index: int, *, train: bool) -> int:
        step_range = self.cfg.train_steps if train else self.cfg.val_steps
        rng = _seeded_rng(self.cfg.seed + (0 if train else 10_000), index)
        return rng.randint(step_range.min_step, step_range.max_step)

    def sample(self, index: int, *, train: bool) -> TaskSample:
        depth = self._choose_depth(index, train=train)
        ex = self._generator(depth, train=train).generate(index)
        return task_sample_from_logic_example(ex, cfg=self.cfg, depth=depth)

    def build_samples(self, n: int, *, train: bool, start_index: int = 0) -> List[TaskSample]:
        return [self.sample(start_index + i, train=train) for i in range(n)]


def task_sample_from_logic_example(ex: LogicExample, *, cfg: TaskConfig, depth: int) -> TaskSample:
    logic_constants = _join_unnumbered(ex.constants)
    logic_predicates = _join_unnumbered(ex.predicates)
    logic_premises = _join_unnumbered(ex.premises_fol)
    logic_proof = _join_unnumbered(ex.proof_fol)
    logic_conclusion = ex.proof_fol[-1].split(". ", 1)[1].split(" ; ", 1)[0].strip()

    facts, rules = _extract_facts_rules(ex.premises_nl)
    natural_facts = "\n".join(facts)
    natural_rules = "\n".join(rules)
    natural_proof = _join_unnumbered(ex.proof_nl)
    natural_conclusion = ex.proof_nl[-1].split(". ", 1)[1].strip()
    natural_premises = _join_unnumbered(ex.premises_nl)
    natural_theory_numbered = "\n".join(ex.premises_nl)

    logic_block = (
        "<formal>\n"
        "<constants>\n" + logic_constants + "\n</constants>\n"
        "<predicates>\n" + logic_predicates + "\n</predicates>\n"
        "<premises>\n" + logic_premises + "\n</premises>\n"
        "<proof>\n" + logic_proof + "\n</proof>\n"
        "<conclusion>\n" + logic_conclusion + "\n</conclusion>\n"
        "</formal>"
    )
    natural_block = (
        "<natural>\n"
        "<facts>\n" + natural_facts + "\n</facts>\n"
        "<rules>\n" + natural_rules + "\n</rules>\n"
        "<proof>\n" + natural_proof + "\n</proof>\n"
        "<conclusion>\n" + natural_conclusion + "\n</conclusion>\n"
        "</natural>"
    )
    think_block = (
        "<think>\n"
        "<premises>\n" + natural_premises + "\n</premises>\n"
        "<proof>\n" + natural_proof + "\n</proof>\n"
        "<conclusion>\n" + natural_conclusion + "\n</conclusion>\n"
        "</think>"
    )
    formal_block = (
        "<formal>\n"
        "<constants>\n" + logic_constants + "\n</constants>\n"
        "<predicates>\n" + logic_predicates + "\n</predicates>\n"
        "<premises>\n" + logic_premises + "\n</premises>\n"
        "<proof>\n" + logic_proof + "\n</proof>\n"
        "<conclusion>\n" + logic_conclusion + "\n</conclusion>\n"
        "</formal>"
    )

    if cfg.template == TemplateName.LOGIC:
        target_body = logic_block
        question = ex.question_fol
        first_modality = "logic"
        first_lines = ex.constants + ex.predicates + ex.premises_fol
        first_prefix_text = (
            "<formal>\n"
            "<constants>\n" + logic_constants + "\n</constants>\n"
            "<predicates>\n" + logic_predicates + "\n</predicates>\n"
            "<premises>\n" + logic_premises + "\n</premises>\n"
        )
    elif cfg.template == TemplateName.NATURAL:
        target_body = natural_block
        question = ex.question_nl
        first_modality = "natural"
        first_lines = facts + rules
        first_prefix_text = (
            "<natural>\n"
            "<facts>\n" + natural_facts + "\n</facts>\n"
            "<rules>\n" + natural_rules + "\n</rules>\n"
        )
    elif cfg.template == TemplateName.LOGIC_NATURAL:
        target_body = logic_block + "\n\n" + natural_block
        question = ex.question_fol + "\n" + ex.question_nl
        first_modality = "logic"
        first_lines = ex.constants + ex.predicates + ex.premises_fol
        first_prefix_text = (
            "<formal>\n"
            "<constants>\n" + logic_constants + "\n</constants>\n"
            "<predicates>\n" + logic_predicates + "\n</predicates>\n"
            "<premises>\n" + logic_premises + "\n</premises>\n"
        )
    elif cfg.template == TemplateName.NL_EXACT:
        target_body = think_block
        question = ex.question_nl
        first_modality = "natural"
        first_lines = ex.premises_nl
        first_prefix_text = (
            "<think>\n"
            "<premises>\n" + natural_premises + "\n</premises>\n"
        )
    elif cfg.template == TemplateName.FORMAL_THINK:
        target_body = formal_block + "\n\n" + think_block
        question = ex.question_fol + "\n" + ex.question_nl
        first_modality = "logic"
        first_lines = ex.constants + ex.predicates + ex.premises_fol
        first_prefix_text = (
            "<formal>\n"
            "<constants>\n" + logic_constants + "\n</constants>\n"
            "<predicates>\n" + logic_predicates + "\n</predicates>\n"
            "<premises>\n" + logic_premises + "\n</premises>\n"
        )
    elif cfg.template == TemplateName.THINK_FORMAL:
        target_body = think_block + "\n\n" + formal_block
        question = ex.question_nl + "\n" + ex.question_fol
        first_modality = "natural"
        first_lines = ex.premises_nl
        first_prefix_text = (
            "<think>\n"
            "<premises>\n" + natural_premises + "\n</premises>\n"
        )
    else:
        target_body = natural_block + "\n\n" + logic_block
        question = ex.question_nl + "\n" + ex.question_fol
        first_modality = "natural"
        first_lines = facts + rules
        first_prefix_text = (
            "<natural>\n"
            "<facts>\n" + natural_facts + "\n</facts>\n"
            "<rules>\n" + natural_rules + "\n</rules>\n"
        )

    prefill_text = ""
    if cfg.prefill == PrefillMode.GOLD:
        prefill_text = "\nGold prefix (copy exactly, then continue):\n" + first_prefix_text

    prompt = (
        "<question>\n"
        f"{natural_theory_numbered}\n"
        f"{ex.question_nl}\n"
        "</question>\n"
        + prefill_text
        + "\n"
    )

    target = target_body + "\n<answer>\n" + ex.answer + "\n</answer>"
    return TaskSample(
        prompt=prompt,
        target=target,
        depth=depth,
        answer=ex.answer,
        logic_constants=logic_constants,
        logic_predicates=logic_predicates,
        logic_premises=logic_premises,
        logic_conclusion=logic_conclusion,
        gold_first_modality_lines=first_lines,
        metadata=dict(ex.metadata),
    )


def task_sample_from_materialized_row(row: dict, *, cfg: TaskConfig) -> TaskSample:
    ex = LogicExample(
        constants=list(row["constants"]),
        predicates=list(row["predicates"]),
        premises_fol=list(row["premises_fol"]),
        premises_nl=list(row["premises_nl"]),
        proof_fol=list(row["proof_fol"]),
        proof_nl=list(row["proof_nl"]),
        question_fol=str(row["question_fol"]),
        question_nl=str(row["question_nl"]),
        answer=str(row["answer"]),
        metadata=dict(row.get("metadata", {})),
    )
    return task_sample_from_logic_example(ex, cfg=cfg, depth=int(row["depth"]))
