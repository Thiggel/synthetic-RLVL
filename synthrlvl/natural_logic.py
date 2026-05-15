from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NaturalProofTranslation:
    proof: str
    parsed_lines: int
    total_lines: int
    errors: tuple[str, ...]

    @property
    def parse_fraction(self) -> float:
        return float(self.parsed_lines / self.total_lines) if self.total_lines else 0.0

    @property
    def fully_parsed(self) -> bool:
        return self.total_lines > 0 and self.parsed_lines == self.total_lines


def _strip_number(text: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", text or "").strip()


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[.]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _constant_map(constants: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in (constants or "").splitlines():
        text = _strip_number(raw)
        if not text:
            continue
        if "=" in text:
            left, right = [part.strip() for part in text.split("=", 1)]
            if left:
                mapping[_norm(left)] = left
            if right:
                mapping[_norm(right)] = left
        else:
            mapping[_norm(text)] = text
    return mapping


def _predicate_map(predicates: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in (predicates or "").splitlines():
        text = _strip_number(raw)
        # Current synthetic predicates are rendered as "Ax: x is blue".
        m = re.match(r"^\s*([A-Z])\w*\s*:\s*x\s+is\s+(.+?)\s*$", text)
        if m:
            mapping[_norm(m.group(2))] = m.group(1)
    return mapping


def _assertion_clause(line: str) -> str:
    text = _norm(_strip_number(line))
    if text.startswith("since ") and "," in text:
        text = text.rsplit(",", 1)[1].strip()
    for prefix in ("therefore ", "thus ", "hence ", "so "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def _split_attributes(raw_attrs: str) -> list[str]:
    attrs = _norm(raw_attrs)
    if attrs.startswith("both "):
        attrs = attrs[len("both ") :].strip()
    if " and " in attrs:
        return [_norm(part) for part in attrs.split(" and ") if _norm(part)]
    return [attrs] if attrs else []


def translate_natural_sentence_to_formula(
    sentence: str,
    *,
    constants: str,
    predicates: str,
) -> str | None:
    """Translate one controlled synthetic NL assertion into a FOL formula.

    This is intentionally grammar-bound, not a general NL parser. It supports the
    sentence forms emitted by the synthetic generator, for example:
    "a is teal", "Grace is both kind and alert", and
    "Since Grace is alert, Grace is dry".
    """
    const_by_name = _constant_map(constants)
    pred_by_attr = _predicate_map(predicates)
    clause = _assertion_clause(sentence)
    m = re.match(r"^(?P<entity>.+?)\s+is\s+(?P<attrs>.+)$", clause)
    if not m:
        return None
    entity = const_by_name.get(_norm(m.group("entity")))
    if not entity:
        return None
    formulas: list[str] = []
    for attr in _split_attributes(m.group("attrs")):
        pred = pred_by_attr.get(_norm(attr))
        if not pred:
            return None
        formulas.append(f"{pred}{entity}")
    if not formulas:
        return None
    return " & ".join(formulas)


def translate_natural_proof_to_fol(
    proof_text: str,
    *,
    constants: str,
    predicates: str,
    premise_count: int,
) -> NaturalProofTranslation:
    proof_lines: list[str] = []
    errors: list[str] = []
    parsed = 0
    total = 0
    next_line = int(premise_count) + 1
    for raw in (proof_text or "").splitlines():
        text = raw.strip()
        if not text:
            continue
        total += 1
        formula = translate_natural_sentence_to_formula(text, constants=constants, predicates=predicates)
        if formula is None:
            formula = "INVALID"
            errors.append(f"line {total}: could not translate `{_strip_number(text)}`")
        else:
            parsed += 1
        # Citation-free proof analysis recovers dependencies and ignores the
        # specific justification, but the parser still requires a known token.
        proof_lines.append(f"{next_line}. {formula} ; R")
        next_line += 1
    return NaturalProofTranslation(
        proof="\n".join(proof_lines),
        parsed_lines=parsed,
        total_lines=total,
        errors=tuple(errors),
    )
