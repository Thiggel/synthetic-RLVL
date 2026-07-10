from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from synthrlvl.datasets import (
    PAIRED_DATASET_KINDS,
    DatasetConfig,
    LogicDatasetGenerator,
    LogicExample,
    PairedGeneratorConfig,
    PairedSyntheticGenerator,
)

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


def _line_body(line: str) -> str:
    stripped = line.strip()
    if ". " in stripped:
        return stripped.split(". ", 1)[1].strip()
    return stripped


def _logic_formula(line: str) -> str:
    return _line_body(line).split(" ; ", 1)[0].strip()


def _logic_rule_label(line: str) -> str:
    body = _line_body(line)
    if " ; " not in body:
        return "unknown"
    justification = body.split(" ; ", 1)[1].strip()
    return justification.split(",", 1)[0].strip() or "unknown"


def _stable_int(parts: list[str]) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:16], 16)


def _shuffle_example_lines(lines: List[str], *, ex: LogicExample, cfg: TaskConfig, depth: int, salt: str) -> List[str]:
    out = list(lines)
    if len(out) < 2:
        return out
    meta = ex.metadata or {}
    index = str(meta.get("record_index", meta.get("index", "")))
    seed_value = _stable_int([str(cfg.seed), salt, str(depth), ex.answer, index, "\n".join(lines)])
    rng = random.Random(seed_value)
    rng.shuffle(out)
    return out


def _terse_natural_proof(lines: List[str]) -> str:
    out: List[str] = []
    for line in lines:
        text = _line_body(line)
        text = text.removeprefix("Therefore ").strip()
        text = text.replace("Combining ", "Using ")
        if text and not text.endswith("."):
            text += "."
        out.append(text)
    return "\n".join(out)


def _rule_annotated_natural_proof(proof_nl: List[str], proof_fol: List[str]) -> str:
    out: List[str] = []
    for nl_line, fol_line in zip(proof_nl, proof_fol, strict=False):
        text = _line_body(nl_line).rstrip(".")
        out.append(f"{text}. [rule: {_logic_rule_label(fol_line)}]")
    return "\n".join(out)


def _pseudocode_proof(proof_nl: List[str], proof_fol: List[str]) -> str:
    out: List[str] = []
    for i, (nl_line, fol_line) in enumerate(zip(proof_nl, proof_fol, strict=False), start=1):
        statement = _line_body(nl_line).rstrip(".")
        out.append(f'step_{i}: derive "{statement}" using {_logic_rule_label(fol_line)}.')
    return "\n".join(out)


def _invalid_logic_proof(proof_fol: List[str]) -> str:
    return "\n".join(f"{_logic_formula(line)} ; R" for line in proof_fol)


def _logic_symbol_aliases(ex: LogicExample) -> tuple[dict[str, str], dict[str, str]]:
    const_aliases: dict[str, str] = {}
    pred_aliases: dict[str, str] = {}
    for line in ex.constants:
        body = _line_body(line)
        if "=" not in body:
            continue
        symbol = body.split("=", 1)[0].strip()
        if symbol and symbol[0].islower():
            const_aliases[symbol] = f"c{symbol}"
    for line in ex.predicates:
        body = _line_body(line)
        if not body:
            continue
        head = body.split(":", 1)[0].strip()
        if len(head) == 2 and head[0].isupper() and head[1] == "x":
            pred_aliases[head[0]] = f"P{head[0]}"
    return pred_aliases, const_aliases


def _pad_logic_atoms(formula: str, *, pred_aliases: dict[str, str], const_aliases: dict[str, str]) -> str:
    def repl(match):
        pred, const = match.groups()
        pred_alias = pred_aliases.get(pred, pred)
        const_alias = const_aliases.get(const, const)
        return f"{pred_alias}({const_alias})"

    # Accept both legacy atoms such as Ba and explicit atoms such as B(c18).
    # Padded traces use explicit calls so multi-character aliases stay atomic.
    padded = re.sub(
        r"(?<![A-Za-z0-9_])([A-Z])\(([a-z][A-Za-z0-9_]*)\)",
        repl,
        formula,
    )
    return re.sub(r"(?<![A-Za-z0-9_])([A-Z])([a-z])(?![A-Za-z0-9_(])", repl, padded)


def _pad_logic_line(line: str, *, pred_aliases: dict[str, str], const_aliases: dict[str, str]) -> str:
    body = _line_body(line)
    if " ; " in body:
        formula, justification = body.split(" ; ", 1)
        return f"{_pad_logic_atoms(formula, pred_aliases=pred_aliases, const_aliases=const_aliases)} ; {justification}"
    return _pad_logic_atoms(body, pred_aliases=pred_aliases, const_aliases=const_aliases)


def _pad_logic_constants(lines: List[str], *, const_aliases: dict[str, str]) -> str:
    out: List[str] = []
    for line in lines:
        body = _line_body(line)
        if "=" not in body:
            out.append(body)
            continue
        symbol, desc = body.split("=", 1)
        symbol = symbol.strip()
        out.append(f"{const_aliases.get(symbol, symbol)} = {desc.strip()}")
    return "\n".join(out)


def _pad_logic_predicates(lines: List[str], *, pred_aliases: dict[str, str]) -> str:
    out: List[str] = []
    for line in lines:
        body = _line_body(line)
        if ":" not in body:
            out.append(body)
            continue
        head, desc = body.split(":", 1)
        head = head.strip()
        if len(head) == 2 and head[0].isupper() and head[1] == "x":
            out.append(f"{pred_aliases.get(head[0], head[0])}(x):{desc}")
        else:
            out.append(body)
    return "\n".join(out)


def _word_symbol_aliases(ex: LogicExample) -> tuple[dict[str, str], dict[str, str]]:
    const_aliases: dict[str, str] = {}
    pred_aliases: dict[str, str] = {}
    used_predicates: set[str] = set()

    for line in ex.constants:
        body = _line_body(line)
        if "=" not in body:
            continue
        symbol = body.split("=", 1)[0].strip()
        if symbol and symbol[0].islower():
            const_aliases[symbol] = symbol

    for line in ex.predicates:
        body = _line_body(line)
        if ":" not in body:
            continue
        head, desc = body.split(":", 1)
        head = head.strip()
        if not (len(head) == 2 and head[0].isupper() and head[1] == "x"):
            continue
        words = re.findall(r"[A-Za-z0-9]+", desc.strip().lower())
        stem = words[-1] if words else head[0].lower()
        name = re.sub(r"[^A-Za-z0-9_]", "", stem.title().replace("_", ""))
        if not name or not name[0].isalpha():
            name = f"State{name or head[0]}"
        if name in used_predicates:
            name = f"{name}_{head[0]}"
        used_predicates.add(name)
        pred_aliases[head[0]] = name

    return pred_aliases, const_aliases


def _wordify_logic_atoms(formula: str, *, pred_aliases: dict[str, str], const_aliases: dict[str, str]) -> str:
    def repl(match):
        pred, const = match.groups()
        pred_alias = pred_aliases.get(pred, pred)
        const_alias = const_aliases.get(const, const)
        return f"{pred_alias}({const_alias})"

    wordified = re.sub(
        r"(?<![A-Za-z0-9_])([A-Z])\(([a-z][A-Za-z0-9_]*)\)",
        repl,
        formula,
    )
    return re.sub(r"(?<![A-Za-z0-9_])([A-Z])([a-z])(?![A-Za-z0-9_(])", repl, wordified)


def _wordify_logic_line(line: str, *, pred_aliases: dict[str, str], const_aliases: dict[str, str]) -> str:
    body = _line_body(line)
    if " ; " in body:
        formula, justification = body.split(" ; ", 1)
        return f"{_wordify_logic_atoms(formula, pred_aliases=pred_aliases, const_aliases=const_aliases)} ; {justification}"
    return _wordify_logic_atoms(body, pred_aliases=pred_aliases, const_aliases=const_aliases)


def _wordify_logic_constants(lines: List[str], *, const_aliases: dict[str, str]) -> str:
    out: List[str] = []
    for line in lines:
        body = _line_body(line)
        if "=" not in body:
            out.append(body)
            continue
        symbol, desc = body.split("=", 1)
        symbol = symbol.strip()
        out.append(f"{const_aliases.get(symbol, symbol)} = {desc.strip()}")
    return "\n".join(out)


def _wordify_logic_predicates(lines: List[str], *, pred_aliases: dict[str, str]) -> str:
    out: List[str] = []
    for line in lines:
        body = _line_body(line)
        if ":" not in body:
            out.append(body)
            continue
        head, desc = body.split(":", 1)
        head = head.strip()
        if len(head) == 2 and head[0].isupper() and head[1] == "x":
            out.append(f"{pred_aliases.get(head[0], head[0])}(x):{desc}")
        else:
            out.append(body)
    return "\n".join(out)


def _conditioned_prompt_prefix(template: TemplateName) -> str:
    if template == TemplateName.CONDITIONED_LOGIC:
        return "<reasoning_mode>\nformal_logic\n</reasoning_mode>\n"
    if template == TemplateName.CONDITIONED_NL:
        return "<reasoning_mode>\nnatural_language\n</reasoning_mode>\n"
    return ""


def _extract_facts_rules(premises_nl: List[str]) -> tuple[List[str], List[str]]:
    facts: List[str] = []
    rules: List[str] = []
    for raw in premises_nl:
        text = raw.split(". ", 1)[1].strip() if ". " in raw else raw.strip()
        if text.startswith(("All things", "For ", "If ")):
            rules.append(text)
        else:
            facts.append(text)
    return facts, rules


class TaskBuilder:
    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        self._gens: Dict[tuple[str, int, bool, float, str], LogicDatasetGenerator | PairedSyntheticGenerator] = {}

    def _generator(self, depth: int, *, train: bool) -> LogicDatasetGenerator | PairedSyntheticGenerator:
        shortcut_rate = self.cfg.shortcut_rate
        if self.cfg.difficulty == "hard_fsa_schema" and not train:
            # Eval is intentionally shortcut-neutral; train may be shortcut-rich.
            shortcut_rate = 0.0
        key = (self.cfg.difficulty, depth, train, shortcut_rate, self.cfg.shortcut_kind)
        if key not in self._gens:
            if self.cfg.difficulty in PAIRED_DATASET_KINDS:
                self._gens[key] = PairedSyntheticGenerator(
                    PairedGeneratorConfig(
                        kind=self.cfg.difficulty,  # type: ignore[arg-type]
                        depth=depth,
                        seed=self.cfg.seed,
                        branching_factor=int(self.cfg.branching_factor or 4),
                        distractor_ratio=float(self.cfg.distractor_ratio),
                    )
                )
            else:
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
                    shortcut_kind=self.cfg.shortcut_kind,
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
    pred_aliases, const_aliases = _logic_symbol_aliases(ex)
    padded_logic_constants = _pad_logic_constants(ex.constants, const_aliases=const_aliases)
    padded_logic_predicates = _pad_logic_predicates(ex.predicates, pred_aliases=pred_aliases)
    padded_logic_premises = "\n".join(
        _pad_logic_line(line, pred_aliases=pred_aliases, const_aliases=const_aliases)
        for line in ex.premises_fol
    )
    padded_logic_proof = "\n".join(
        _pad_logic_line(line, pred_aliases=pred_aliases, const_aliases=const_aliases)
        for line in ex.proof_fol
    )
    padded_logic_conclusion = _pad_logic_atoms(
        logic_conclusion,
        pred_aliases=pred_aliases,
        const_aliases=const_aliases,
    )
    word_pred_aliases, word_const_aliases = _word_symbol_aliases(ex)
    word_logic_constants = _wordify_logic_constants(ex.constants, const_aliases=word_const_aliases)
    word_logic_predicates = _wordify_logic_predicates(ex.predicates, pred_aliases=word_pred_aliases)
    word_logic_premises = "\n".join(
        _wordify_logic_line(line, pred_aliases=word_pred_aliases, const_aliases=word_const_aliases)
        for line in ex.premises_fol
    )
    word_logic_proof = "\n".join(
        _wordify_logic_line(line, pred_aliases=word_pred_aliases, const_aliases=word_const_aliases)
        for line in ex.proof_fol
    )
    word_logic_conclusion = _wordify_logic_atoms(
        logic_conclusion,
        pred_aliases=word_pred_aliases,
        const_aliases=word_const_aliases,
    )

    facts, rules = _extract_facts_rules(ex.premises_nl)
    natural_facts = "\n".join(facts)
    natural_rules = "\n".join(rules)
    natural_proof = _join_unnumbered(ex.proof_nl)
    terse_natural_proof = _terse_natural_proof(ex.proof_nl)
    rule_annotated_natural_proof = _rule_annotated_natural_proof(ex.proof_nl, ex.proof_fol)
    pseudocode_proof = _pseudocode_proof(ex.proof_nl, ex.proof_fol)
    shuffled_logic_proof = _join_unnumbered(
        _shuffle_example_lines(ex.proof_fol, ex=ex, cfg=cfg, depth=depth, salt="shuffled_logic")
    )
    shuffled_natural_proof = _join_unnumbered(
        _shuffle_example_lines(ex.proof_nl, ex=ex, cfg=cfg, depth=depth, salt="shuffled_nl")
    )
    invalid_logic_proof = _invalid_logic_proof(ex.proof_fol)
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
    padded_logic_block = (
        "<formal>\n"
        "<constants>\n" + padded_logic_constants + "\n</constants>\n"
        "<predicates>\n" + padded_logic_predicates + "\n</predicates>\n"
        "<premises>\n" + padded_logic_premises + "\n</premises>\n"
        "<proof>\n" + padded_logic_proof + "\n</proof>\n"
        "<conclusion>\n" + padded_logic_conclusion + "\n</conclusion>\n"
        "</formal>"
    )
    word_logic_block = (
        "<formal>\n"
        "<constants>\n" + word_logic_constants + "\n</constants>\n"
        "<predicates>\n" + word_logic_predicates + "\n</predicates>\n"
        "<premises>\n" + word_logic_premises + "\n</premises>\n"
        "<proof>\n" + word_logic_proof + "\n</proof>\n"
        "<conclusion>\n" + word_logic_conclusion + "\n</conclusion>\n"
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
    terse_think_block = (
        "<think>\n"
        "<premises>\n" + natural_premises + "\n</premises>\n"
        "<proof>\n" + terse_natural_proof + "\n</proof>\n"
        "<conclusion>\n" + natural_conclusion + "\n</conclusion>\n"
        "</think>"
    )
    rule_annotated_think_block = (
        "<think>\n"
        "<premises>\n" + natural_premises + "\n</premises>\n"
        "<proof>\n" + rule_annotated_natural_proof + "\n</proof>\n"
        "<conclusion>\n" + natural_conclusion + "\n</conclusion>\n"
        "</think>"
    )
    pseudocode_think_block = (
        "<think>\n"
        "<premises>\n" + natural_premises + "\n</premises>\n"
        "<proof>\n" + pseudocode_proof + "\n</proof>\n"
        "<conclusion>\n" + natural_conclusion + "\n</conclusion>\n"
        "</think>"
    )
    shuffled_think_block = (
        "<think>\n"
        "<premises>\n" + natural_premises + "\n</premises>\n"
        "<proof>\n" + shuffled_natural_proof + "\n</proof>\n"
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
    shuffled_logic_block = (
        "<formal>\n"
        "<constants>\n" + logic_constants + "\n</constants>\n"
        "<predicates>\n" + logic_predicates + "\n</predicates>\n"
        "<premises>\n" + logic_premises + "\n</premises>\n"
        "<proof>\n" + shuffled_logic_proof + "\n</proof>\n"
        "<conclusion>\n" + logic_conclusion + "\n</conclusion>\n"
        "</formal>"
    )
    invalid_logic_block = (
        "<formal>\n"
        "<constants>\n" + logic_constants + "\n</constants>\n"
        "<predicates>\n" + logic_predicates + "\n</predicates>\n"
        "<premises>\n" + logic_premises + "\n</premises>\n"
        "<proof>\n" + invalid_logic_proof + "\n</proof>\n"
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
    elif cfg.template == TemplateName.LOGIC_SYMBOL_PADDED:
        target_body = padded_logic_block
        question = ex.question_fol
        first_modality = "logic"
        first_lines = ex.constants + ex.predicates + ex.premises_fol
        first_prefix_text = (
            "<formal>\n"
            "<constants>\n" + padded_logic_constants + "\n</constants>\n"
            "<predicates>\n" + padded_logic_predicates + "\n</predicates>\n"
            "<premises>\n" + padded_logic_premises + "\n</premises>\n"
        )
    elif cfg.template == TemplateName.LOGIC_WORDIFIED:
        target_body = word_logic_block
        question = ex.question_fol
        first_modality = "logic"
        first_lines = ex.constants + ex.predicates + ex.premises_fol
        first_prefix_text = (
            "<formal>\n"
            "<constants>\n" + word_logic_constants + "\n</constants>\n"
            "<predicates>\n" + word_logic_predicates + "\n</predicates>\n"
            "<premises>\n" + word_logic_premises + "\n</premises>\n"
        )
    elif cfg.template == TemplateName.CONDITIONED_LOGIC:
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
    elif cfg.template == TemplateName.CONDITIONED_NL:
        target_body = think_block
        question = ex.question_nl
        first_modality = "natural"
        first_lines = ex.premises_nl
        first_prefix_text = (
            "<think>\n"
            "<premises>\n" + natural_premises + "\n</premises>\n"
        )
    elif cfg.template in (
        TemplateName.TERSE_NL,
        TemplateName.RULE_ANNOTATED_NL,
        TemplateName.PSEUDOCODE,
        TemplateName.SHUFFLED_NL,
    ):
        if cfg.template == TemplateName.TERSE_NL:
            target_body = terse_think_block
        elif cfg.template == TemplateName.RULE_ANNOTATED_NL:
            target_body = rule_annotated_think_block
        elif cfg.template == TemplateName.PSEUDOCODE:
            target_body = pseudocode_think_block
        else:
            target_body = shuffled_think_block
        question = ex.question_nl
        first_modality = "natural"
        first_lines = ex.premises_nl
        first_prefix_text = (
            "<think>\n"
            "<premises>\n" + natural_premises + "\n</premises>\n"
        )
    elif cfg.template in (TemplateName.SHUFFLED_LOGIC, TemplateName.INVALID_LOGIC):
        target_body = shuffled_logic_block if cfg.template == TemplateName.SHUFFLED_LOGIC else invalid_logic_block
        question = ex.question_fol
        first_modality = "logic"
        first_lines = ex.constants + ex.predicates + ex.premises_fol
        first_prefix_text = (
            "<formal>\n"
            "<constants>\n" + logic_constants + "\n</constants>\n"
            "<predicates>\n" + logic_predicates + "\n</predicates>\n"
            "<premises>\n" + logic_premises + "\n</premises>\n"
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
        _conditioned_prompt_prefix(cfg.template)
        +
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
        logic_constants=(
            padded_logic_constants
            if cfg.template == TemplateName.LOGIC_SYMBOL_PADDED
            else word_logic_constants
            if cfg.template == TemplateName.LOGIC_WORDIFIED
            else logic_constants
        ),
        logic_predicates=(
            padded_logic_predicates
            if cfg.template == TemplateName.LOGIC_SYMBOL_PADDED
            else word_logic_predicates
            if cfg.template == TemplateName.LOGIC_WORDIFIED
            else logic_predicates
        ),
        logic_premises=(
            padded_logic_premises
            if cfg.template == TemplateName.LOGIC_SYMBOL_PADDED
            else word_logic_premises
            if cfg.template == TemplateName.LOGIC_WORDIFIED
            else logic_premises
        ),
        logic_conclusion=(
            padded_logic_conclusion
            if cfg.template == TemplateName.LOGIC_SYMBOL_PADDED
            else word_logic_conclusion
            if cfg.template == TemplateName.LOGIC_WORDIFIED
            else logic_conclusion
        ),
        gold_first_modality_lines=first_lines,
        metadata=dict(ex.metadata),
    )


def task_sample_from_materialized_row(row: dict, *, cfg: TaskConfig) -> TaskSample:
    metadata = dict(row.get("metadata", {}))
    if "record_index" in row:
        metadata["record_index"] = int(row["record_index"])
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
        metadata=metadata,
    )
    return task_sample_from_logic_example(ex, cfg=cfg, depth=int(row["depth"]))
