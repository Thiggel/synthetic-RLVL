from __future__ import annotations

import re
from dataclasses import dataclass

from logic_engine import LogicEngine

from .types import PrefillMode, RewardSchema, TemplateName


TAG_BLOCK = re.compile(r"<(?P<tag>[a-z_]+)>\s*(?P<body>.*?)\s*</\1>", flags=re.DOTALL | re.IGNORECASE)
STRICT_LOGIC_OUTPUT = re.compile(
    r"\s*<formal>\s*"
    r"<constants>\s*.*?\s*</constants>\s*"
    r"<predicates>\s*.*?\s*</predicates>\s*"
    r"<premises>\s*.*?\s*</premises>\s*"
    r"<proof>\s*.*?\s*</proof>\s*"
    r"<conclusion>\s*.*?\s*</conclusion>\s*"
    r"</formal>\s*"
    r"<answer>\s*.*?\s*</answer>\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_heading_section(text: str, heading: str, *, stop_at: list[str]) -> str:
    if not text:
        return ""
    stop_group = "|".join(re.escape(item) for item in stop_at)
    pattern = re.compile(
        rf"(?is)(?:^|\n)\s*{re.escape(heading)}\s*[:\-]?\s*\n?(.*?)\s*(?=(?:\n\s*(?:{stop_group})\s*[:\-]?\s*)|\Z)"
    )
    m = pattern.search(text)
    return (m.group(1) if m else "").strip()


def split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _normalize_answer_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _is_answer_match(pred_answer: str, gold_answer: str) -> bool:
    pred_norm = _normalize_answer_text(pred_answer)
    gold_norm = _normalize_answer_text(gold_answer)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    # Accept natural phrasing such as "Yara is sparse." for gold answer "sparse".
    return re.search(rf"(?:^|\s){re.escape(gold_norm)}(?:\s|$)", pred_norm) is not None


@dataclass(frozen=True)
class EvalResult:
    syntactic: float
    format_ok: float
    correct: float
    valid: float
    line_match: float


class OutputEvaluator:
    def __init__(self):
        self.engine = LogicEngine()

    @staticmethod
    def _has_strict_logic_layout(text: str) -> bool:
        return STRICT_LOGIC_OUTPUT.fullmatch(text or "") is not None

    @staticmethod
    def _logic_block_tag(template: TemplateName) -> str:
        # All logic-structured outputs now use <formal> for consistency.
        return "formal"

    @staticmethod
    def _natural_block_tag(template: TemplateName) -> str:
        if template in (TemplateName.NL_EXACT, TemplateName.FORMAL_THINK, TemplateName.THINK_FORMAL):
            return "think"
        return "natural"

    @staticmethod
    def _has_logic_structure(text: str, block_tag: str) -> bool:
        logic = extract_tag(text, block_tag)
        if not logic:
            return False
        return all(extract_tag(logic, tag) for tag in ["constants", "predicates", "premises", "proof", "conclusion"])

    @staticmethod
    def _has_natural_structure(text: str, block_tag: str, uses_premises_rules: bool) -> bool:
        natural = extract_tag(text, block_tag)
        if not natural:
            return False
        if uses_premises_rules:
            required = ["premises", "proof", "conclusion"]
        else:
            required = ["facts", "rules", "proof", "conclusion"]
        return all(extract_tag(natural, tag) for tag in required)

    @staticmethod
    def _extract_logic_components(text: str, block_tag: str) -> tuple[str, str, str]:
        logic = extract_tag(text, block_tag)
        source = logic if logic else (text or "")
        premises = extract_tag(source, "premises")
        proof = extract_tag(source, "proof")
        conclusion = extract_tag(source, "conclusion")
        if premises and proof and conclusion:
            return premises, proof, conclusion
        # Fallback for generations that use section headings instead of XML tags.
        premises = premises or _extract_heading_section(
            source,
            "premises",
            stop_at=["proof", "conclusion", "answer"],
        )
        proof = proof or _extract_heading_section(
            source,
            "proof",
            stop_at=["conclusion", "answer"],
        )
        conclusion = conclusion or _extract_heading_section(
            source,
            "conclusion",
            stop_at=["answer", "<answer>"],
        )
        return premises, proof, conclusion

    def evaluate(
        self,
        output_text: str,
        *,
        template: TemplateName,
        gold_answer: str,
        gold_logic_premises: str,
        gold_logic_conclusion: str,
        prefill: PrefillMode,
        gold_first_modality_lines: list[str],
    ) -> EvalResult:
        answer = extract_tag(output_text, "answer")
        correct = float(_is_answer_match(answer, gold_answer))

        wants_logic = template in (
            TemplateName.LOGIC,
            TemplateName.LOGIC_NATURAL,
            TemplateName.NATURAL_LOGIC,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
        )
        wants_natural = template in (
            TemplateName.NATURAL,
            TemplateName.LOGIC_NATURAL,
            TemplateName.NATURAL_LOGIC,
            TemplateName.NL_EXACT,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
        )
        logic_tag = self._logic_block_tag(template)
        natural_tag = self._natural_block_tag(template)
        natural_uses_premises_rules = template in (TemplateName.NL_EXACT, TemplateName.FORMAL_THINK, TemplateName.THINK_FORMAL)

        format_ok = 1.0
        # For pure logic format reward, require canonical tag order with no extra content outside tags.
        if template == TemplateName.LOGIC and not self._has_strict_logic_layout(output_text):
            format_ok = 0.0
        if wants_logic and not self._has_logic_structure(output_text, logic_tag):
            format_ok = 0.0
        if wants_natural and not self._has_natural_structure(output_text, natural_tag, natural_uses_premises_rules):
            format_ok = 0.0
        if not answer:
            format_ok = 0.0

        syntactic = 0.0
        valid = 0.0
        if wants_logic:
            premises, proof, conclusion = self._extract_logic_components(output_text, logic_tag)
            if premises and proof and conclusion:
                report = self.engine.analyze_proof(premises=premises, conclusion=conclusion, proof=proof)
                syntactic = float(
                    bool(report.lines)
                    and all(p.syntax_valid for p in report.premises)
                    and all(line.syntax_valid for line in report.lines)
                )
                valid = float(report.ok)

        line_match = 0.0
        if prefill == PrefillMode.LINE_REWARD and gold_first_modality_lines:
            natural_first_templates = (TemplateName.NATURAL, TemplateName.NATURAL_LOGIC, TemplateName.NL_EXACT, TemplateName.THINK_FORMAL)
            block = extract_tag(output_text, natural_tag) if template in natural_first_templates else extract_tag(output_text, logic_tag)
            pred_lines = split_lines(block)
            wanted = [ln.strip() for ln in gold_first_modality_lines if ln.strip()]
            hits = 0
            for ln in wanted:
                norm = ln.split(". ", 1)[1] if ". " in ln else ln
                if norm in pred_lines:
                    hits += 1
            line_match = hits / max(1, len(wanted))

        return EvalResult(syntactic=syntactic, format_ok=format_ok, correct=correct, valid=valid, line_match=line_match)


class RewardComputer:
    def __init__(self, evaluator: OutputEvaluator):
        self.evaluator = evaluator

    def _line_valid_fraction(self, output_text: str, *, template: TemplateName) -> float:
        wants_logic = template in (
            TemplateName.LOGIC,
            TemplateName.LOGIC_NATURAL,
            TemplateName.NATURAL_LOGIC,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
        )
        if not wants_logic:
            return 0.0
        logic_tag = self.evaluator._logic_block_tag(template)
        premises, proof, conclusion = self.evaluator._extract_logic_components(output_text, logic_tag)
        if not premises or not proof or not conclusion:
            return 0.0
        try:
            report = self.evaluator.engine.analyze_proof(premises=premises, conclusion=conclusion, proof=proof)
            total = len(report.lines)
            if total == 0:
                return 0.0
            valid = sum(1 for line in report.lines if line.valid)
            return float(valid / total)
        except Exception:
            return 0.0

    def reward(
        self,
        output_text: str,
        *,
        schema: RewardSchema,
        template: TemplateName,
        gold_answer: str,
        gold_logic_premises: str,
        gold_logic_conclusion: str,
        prefill: PrefillMode,
        gold_first_modality_lines: list[str],
    ) -> tuple[float, EvalResult]:
        m = self.evaluator.evaluate(
            output_text,
            template=template,
            gold_answer=gold_answer,
            gold_logic_premises=gold_logic_premises,
            gold_logic_conclusion=gold_logic_conclusion,
            prefill=prefill,
            gold_first_modality_lines=gold_first_modality_lines,
        )

        line_valid = None
        if schema in {
            RewardSchema.CORRECT_PLUS_LINE_VALID_PLUS_0P1_FORMAT,
            RewardSchema.CORRECT_TIMES_LINE_VALID_PLUS_0P1_FORMAT,
        }:
            line_valid = self._line_valid_fraction(output_text, template=template)

        if schema == RewardSchema.CORRECT_PLUS_0P1_FORMAT:
            value = m.correct + 0.1 * m.format_ok
        elif schema == RewardSchema.INDICATOR_CORRECT_AND_FORMAT:
            value = float(m.correct > 0 and m.format_ok > 0)
        elif schema == RewardSchema.CORRECT_PLUS_VALID_PLUS_0P1_FORMAT:
            value = m.correct + m.valid + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_PLUS_LINE_VALID_PLUS_0P1_FORMAT:
            value = m.correct + float(line_valid or 0.0) + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_TIMES_VALID_PLUS_0P1_FORMAT:
            value = (m.correct * m.valid) + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_TIMES_LINE_VALID_PLUS_0P1_FORMAT:
            value = (m.correct * float(line_valid or 0.0)) + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_PLUS_0P75_VALID_PLUS_0P1_FORMAT:
            value = m.correct + 0.75 * m.valid + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_PLUS_0P5_VALID_PLUS_0P1_FORMAT:
            value = m.correct + 0.5 * m.valid + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_PLUS_0P25_VALID_PLUS_0P1_FORMAT:
            value = m.correct + 0.25 * m.valid + 0.1 * m.format_ok
        elif schema == RewardSchema.INDICATOR_ALL:
            value = float(m.correct > 0 and m.valid > 0 and m.format_ok > 0)
        else:
            raise ValueError(f"Unknown schema: {schema}")

        if prefill == PrefillMode.LINE_REWARD:
            value += m.line_match
        return value, m
