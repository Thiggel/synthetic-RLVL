from __future__ import annotations

import re
from dataclasses import dataclass

from logic_engine import LogicEngine

from .natural_logic import translate_natural_proof_to_fol
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
    # Accept a single-line natural phrasing such as "Yara is sparse." for
    # gold answer "sparse", but never credit a list that merely contains the
    # gold answer among several alternatives.
    if len(split_lines(pred_answer)) != 1:
        return False
    return (
        re.fullmatch(
            rf"(?:the answer (?:is|equals)|.+ (?:is|are|equals)) {re.escape(gold_norm)}",
            pred_norm,
        )
        is not None
    )


@dataclass(frozen=True)
class EvalResult:
    syntactic: float
    format_ok: float
    correct: float
    valid: float
    citation_free_valid: float
    grounded_valid: float
    citation_free_grounded_valid: float
    nl_logic_parse: float
    nl_logic_citation_free_valid: float
    nl_logic_line_valid_fraction: float
    nl_logic_valid_prefix_fraction: float
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
    def _has_unambiguous_logic_declarations(text: str, block_tag: str) -> bool:
        logic = extract_tag(text, block_tag)
        source = logic if logic else (text or "")
        has_constants_block = re.search(
            r"<constants>\s*.*?\s*</constants>", source, flags=re.DOTALL | re.IGNORECASE
        ) is not None
        has_predicates_block = re.search(
            r"<predicates>\s*.*?\s*</predicates>", source, flags=re.DOTALL | re.IGNORECASE
        ) is not None
        if not has_constants_block and not has_predicates_block:
            return True
        if not has_constants_block or not has_predicates_block:
            return False
        constants = split_lines(extract_tag(source, "constants"))
        predicates = split_lines(extract_tag(source, "predicates"))

        constant_names: list[str] = []
        for line in constants:
            match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*.+?\s*", line)
            if match is None:
                return False
            constant_names.append(match.group(1))

        predicate_names: list[str] = []
        for line in predicates:
            match = re.fullmatch(
                r"\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:x|\(\s*x\s*\))\s*:\s*.+?\s*",
                line,
                flags=re.IGNORECASE,
            )
            if match is None:
                return False
            predicate_names.append(match.group(1))

        return (
            len(constant_names) == len(set(constant_names))
            and len(predicate_names) == len(set(predicate_names))
        )

    @staticmethod
    def _natural_block_tag(template: TemplateName) -> str:
        if template in (
            TemplateName.NL_EXACT,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
            TemplateName.TERSE_NL,
            TemplateName.RULE_ANNOTATED_NL,
            TemplateName.PSEUDOCODE,
            TemplateName.SHUFFLED_NL,
            TemplateName.CONDITIONED_NL,
        ):
            return "think"
        return "natural"

    @staticmethod
    def _has_logic_structure(text: str, block_tag: str) -> bool:
        logic = extract_tag(text, block_tag)
        if not logic:
            return False
        has_header_tags = all(
            re.search(rf"<{tag}>\s*.*?\s*</{tag}>", logic, flags=re.DOTALL | re.IGNORECASE)
            for tag in ["constants", "predicates"]
        )
        has_reasoning_tags = all(extract_tag(logic, tag) for tag in ["premises", "proof", "conclusion"])
        return has_header_tags and has_reasoning_tags

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
        gold_logic_constants: str = "",
        gold_logic_predicates: str = "",
        prefill: PrefillMode = PrefillMode.NONE,
        gold_first_modality_lines: list[str] | None = None,
    ) -> EvalResult:
        answer = extract_tag(output_text, "answer")
        correct = float(_is_answer_match(answer, gold_answer))

        wants_logic = template in (
            TemplateName.LOGIC,
            TemplateName.LOGIC_SYMBOL_PADDED,
            TemplateName.LOGIC_WORDIFIED,
            TemplateName.LOGIC_NATURAL,
            TemplateName.NATURAL_LOGIC,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
            TemplateName.SHUFFLED_LOGIC,
            TemplateName.INVALID_LOGIC,
            TemplateName.CONDITIONED_LOGIC,
        )
        wants_natural = template in (
            TemplateName.NATURAL,
            TemplateName.LOGIC_NATURAL,
            TemplateName.NATURAL_LOGIC,
            TemplateName.NL_EXACT,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
            TemplateName.TERSE_NL,
            TemplateName.RULE_ANNOTATED_NL,
            TemplateName.PSEUDOCODE,
            TemplateName.SHUFFLED_NL,
            TemplateName.CONDITIONED_NL,
        )
        logic_tag = self._logic_block_tag(template)
        natural_tag = self._natural_block_tag(template)
        logic_declarations_ok = (
            self._has_unambiguous_logic_declarations(output_text, logic_tag)
            if wants_logic
            else True
        )
        natural_uses_premises_rules = template in (
            TemplateName.NL_EXACT,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
            TemplateName.TERSE_NL,
            TemplateName.RULE_ANNOTATED_NL,
            TemplateName.PSEUDOCODE,
            TemplateName.SHUFFLED_NL,
            TemplateName.CONDITIONED_NL,
        )

        format_ok = 1.0
        # For pure logic format reward, require canonical tag order with no extra content outside tags.
        if template in (
            TemplateName.LOGIC,
            TemplateName.LOGIC_SYMBOL_PADDED,
            TemplateName.LOGIC_WORDIFIED,
        ) and not self._has_strict_logic_layout(output_text):
            format_ok = 0.0
        if wants_logic and (
            not self._has_logic_structure(output_text, logic_tag)
            or not logic_declarations_ok
        ):
            format_ok = 0.0
        if wants_natural and not self._has_natural_structure(output_text, natural_tag, natural_uses_premises_rules):
            format_ok = 0.0
        if not answer:
            format_ok = 0.0

        syntactic = 0.0
        valid = 0.0
        citation_free_valid = 0.0
        grounded_valid = 0.0
        citation_free_grounded_valid = 0.0
        if wants_logic and logic_declarations_ok:
            premises, proof, conclusion = self._extract_logic_components(output_text, logic_tag)
            if premises and proof and conclusion:
                report = self.engine.analyze_proof(premises=premises, conclusion=conclusion, proof=proof)
                citation_free_report = self.engine.analyze_proof_citation_free(
                    premises=premises, conclusion=conclusion, proof=proof
                )
                syntactic = float(
                    bool(report.lines)
                    and all(p.syntax_valid for p in report.premises)
                    and all(line.syntax_valid for line in report.lines)
                )
                valid = float(report.ok)
                citation_free_valid = float(citation_free_report.ok or report.ok)
                if gold_logic_premises and gold_logic_conclusion:
                    grounded_report = self.engine.analyze_proof(
                        premises=gold_logic_premises,
                        conclusion=gold_logic_conclusion,
                        proof=proof,
                    )
                    citation_free_grounded_report = self.engine.analyze_proof_citation_free(
                        premises=gold_logic_premises,
                        conclusion=gold_logic_conclusion,
                        proof=proof,
                    )
                    grounded_valid = float(grounded_report.ok)
                    citation_free_grounded_valid = float(citation_free_grounded_report.ok or grounded_report.ok)

        nl_logic_parse = 0.0
        nl_logic_citation_free_valid = 0.0
        nl_logic_line_valid_fraction = 0.0
        nl_logic_valid_prefix_fraction = 0.0
        if wants_natural and gold_logic_premises and gold_logic_conclusion and gold_logic_constants:
            natural = extract_tag(output_text, natural_tag)
            source = natural if natural else (output_text or "")
            natural_proof = extract_tag(source, "proof")
            if natural_proof:
                premise_count = len(split_lines(gold_logic_premises))
                translated = translate_natural_proof_to_fol(
                    natural_proof,
                    constants=gold_logic_constants,
                    predicates=gold_logic_predicates,
                    premise_count=premise_count,
                    premises=gold_logic_premises,
                )
                nl_logic_parse = translated.parse_fraction
                if translated.total_lines:
                    strict_report = self.engine.analyze_proof(
                        premises=gold_logic_premises,
                        conclusion=gold_logic_conclusion,
                        proof=translated.proof,
                    )
                    report = self.engine.analyze_proof_citation_free(
                        premises=gold_logic_premises,
                        conclusion=gold_logic_conclusion,
                        proof=translated.proof,
                    )
                    line_report = strict_report if strict_report.ok else report
                    valid_lines = max(
                        sum(1 for line in strict_report.lines if line.valid),
                        sum(1 for line in report.lines if line.valid),
                    )
                    prefix_valid = 0
                    for line in line_report.lines:
                        if not line.valid:
                            break
                        prefix_valid += 1
                    nl_logic_citation_free_valid = float(translated.fully_parsed and (report.ok or strict_report.ok))
                    nl_logic_line_valid_fraction = float(valid_lines / len(line_report.lines)) if line_report.lines else 0.0
                    nl_logic_valid_prefix_fraction = float(prefix_valid / len(line_report.lines)) if line_report.lines else 0.0

        line_match = 0.0
        if prefill == PrefillMode.LINE_REWARD and gold_first_modality_lines:
            natural_first_templates = (
                TemplateName.NATURAL,
                TemplateName.NATURAL_LOGIC,
                TemplateName.NL_EXACT,
                TemplateName.THINK_FORMAL,
                TemplateName.TERSE_NL,
                TemplateName.RULE_ANNOTATED_NL,
                TemplateName.PSEUDOCODE,
                TemplateName.SHUFFLED_NL,
                TemplateName.CONDITIONED_NL,
            )
            block = extract_tag(output_text, natural_tag) if template in natural_first_templates else extract_tag(output_text, logic_tag)
            pred_lines = split_lines(block)
            wanted = [ln.strip() for ln in gold_first_modality_lines if ln.strip()]
            hits = 0
            for ln in wanted:
                norm = ln.split(". ", 1)[1] if ". " in ln else ln
                if norm in pred_lines:
                    hits += 1
            line_match = hits / max(1, len(wanted))

        return EvalResult(
            syntactic=syntactic,
            format_ok=format_ok,
            correct=correct,
            valid=valid,
            citation_free_valid=citation_free_valid,
            grounded_valid=grounded_valid,
            citation_free_grounded_valid=citation_free_grounded_valid,
            nl_logic_parse=nl_logic_parse,
            nl_logic_citation_free_valid=nl_logic_citation_free_valid,
            nl_logic_line_valid_fraction=nl_logic_line_valid_fraction,
            nl_logic_valid_prefix_fraction=nl_logic_valid_prefix_fraction,
            line_match=line_match,
        )


class RewardComputer:
    def __init__(self, evaluator: OutputEvaluator):
        self.evaluator = evaluator

    def _line_valid_fraction(self, output_text: str, *, template: TemplateName, citation_free: bool = False) -> float:
        wants_logic = template in (
            TemplateName.LOGIC,
            TemplateName.LOGIC_SYMBOL_PADDED,
            TemplateName.LOGIC_WORDIFIED,
            TemplateName.LOGIC_NATURAL,
            TemplateName.NATURAL_LOGIC,
            TemplateName.FORMAL_THINK,
            TemplateName.THINK_FORMAL,
            TemplateName.SHUFFLED_LOGIC,
            TemplateName.INVALID_LOGIC,
            TemplateName.CONDITIONED_LOGIC,
        )
        if not wants_logic:
            return 0.0
        logic_tag = self.evaluator._logic_block_tag(template)
        premises, proof, conclusion = self.evaluator._extract_logic_components(output_text, logic_tag)
        if not premises or not proof or not conclusion:
            return 0.0
        try:
            if citation_free:
                report = self.evaluator.engine.analyze_proof_citation_free(
                    premises=premises, conclusion=conclusion, proof=proof
                )
            else:
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
        citation_free_line_valid = None
        if schema in {
            RewardSchema.CORRECT_PLUS_CITATION_FREE_LINE_VALID_PLUS_0P1_FORMAT,
            RewardSchema.CORRECT_TIMES_CITATION_FREE_LINE_VALID_PLUS_0P1_FORMAT,
            RewardSchema.CITATION_FREE_LINE_VALID_PLUS_CORRECT_IF_FULL_VALID_PLUS_0P1_FORMAT,
        }:
            citation_free_line_valid = self._line_valid_fraction(output_text, template=template, citation_free=True)

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
        elif schema == RewardSchema.CORRECT_PLUS_CITATION_FREE_VALID_PLUS_0P1_FORMAT:
            value = m.correct + m.citation_free_valid + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_PLUS_CITATION_FREE_LINE_VALID_PLUS_0P1_FORMAT:
            value = m.correct + float(citation_free_line_valid or 0.0) + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_TIMES_CITATION_FREE_VALID_PLUS_0P1_FORMAT:
            value = (m.correct * m.citation_free_valid) + 0.1 * m.format_ok
        elif schema == RewardSchema.CORRECT_TIMES_CITATION_FREE_LINE_VALID_PLUS_0P1_FORMAT:
            value = (m.correct * float(citation_free_line_valid or 0.0)) + 0.1 * m.format_ok
        elif schema == RewardSchema.CITATION_FREE_LINE_VALID_PLUS_CORRECT_IF_FULL_VALID_PLUS_0P1_FORMAT:
            value = (
                float(citation_free_line_valid or 0.0)
                + (m.correct if m.citation_free_valid > 0 else 0.0)
                + 0.1 * m.format_ok
            )
        elif schema == RewardSchema.INDICATOR_CORRECT_AND_CITATION_FREE_VALID_PLUS_0P1_FORMAT:
            value = float(m.correct > 0 and m.citation_free_valid > 0) + 0.1 * m.format_ok
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
