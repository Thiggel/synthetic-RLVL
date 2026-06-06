from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


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


def _as_lines(text: str | Iterable[str] | None) -> list[str]:
    if text is None:
        return []
    if isinstance(text, str):
        return text.splitlines()
    return [str(item) for item in text]


def _constant_map(constants: str | Iterable[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in _as_lines(constants):
        text = _strip_number(raw)
        if not text:
            continue
        if "=" in text:
            left, right = [part.strip() for part in text.split("=", 1)]
            if left:
                mapping[_norm(left)] = left
            if right:
                mapping[_norm(right)] = left
                for variant in _igsm_quantity_variants(right):
                    mapping[_norm(variant)] = left
        else:
            mapping[_norm(text)] = text
    return mapping


def _igsm_quantity_variants(quantity: str) -> list[str]:
    text = (quantity or "").strip()
    variants = [text]
    for prefix in ("the number of each ", "number of each "):
        if text.lower().startswith(prefix):
            variants.append(text[len(prefix) :].strip())
    return [variant for variant in variants if variant]


def _predicate_map(predicates: str | Iterable[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in _as_lines(predicates):
        text = _strip_number(raw)
        # Synthetic predicates are rendered as "Ax: x is blue" for legacy
        # single-letter names and "P0(x): x is blue" for extended names.
        m = re.match(r"^\s*([A-Z][A-Za-z0-9_]*)\(\s*x\s*\)\s*:\s*x\s+is\s+(.+?)\s*$", text)
        if not m:
            m = re.match(r"^\s*([A-Z])x\s*:\s*x\s+is\s+(.+?)\s*$", text)
        if m:
            mapping[_norm(m.group(2))] = m.group(1)
    return mapping


def _predicate_atom(predicate: str, constant: str) -> str:
    if len(predicate) == 1 and predicate.isalpha():
        return f"{predicate}{constant}"
    return f"{predicate}({constant})"


def _assertion_clause(line: str) -> str:
    line = _unwrap_controlled_trace_line(line)
    text = _norm(_strip_number(line))
    if text.startswith("since ") and "," in text:
        text = text.rsplit(",", 1)[1].strip()
    for prefix in ("therefore ", "thus ", "hence ", "so "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def _unwrap_controlled_trace_line(line: str) -> str:
    """Remove wrappers added by controlled trace ablations before NL parsing."""
    text = _strip_number(line)
    m = re.match(
        r'(?is)^\s*step[_\s-]*\d+\s*:\s*derive\s+["“](?P<statement>.+?)["”]\s+using\s+.+?\s*$',
        text,
    )
    if m:
        return m.group("statement").strip()
    text = re.sub(r"(?is)\s*\[\s*rule\s*:\s*[^\]]+\]\s*$", "", text).strip()
    return text


def _proof_premise_map(premises: str | Iterable[str] | None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    next_idx = 1
    for raw in _as_lines(premises):
        text = _strip_number(raw)
        if not text:
            continue
        m = re.match(r"^\s*(\d+)\.\s*(.+)$", str(raw).strip())
        line_number = int(m.group(1)) if m else next_idx
        formula = m.group(2).strip() if m else text
        mapping[_canonical_formula_text(formula)] = line_number
        next_idx = max(next_idx + 1, line_number + 1)
    return mapping


def _canonical_formula_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _official_var_name(var: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", var.strip())
    if cleaned.startswith("v_"):
        return cleaned
    return f"v_{cleaned}"


def _bare_igsm_var(var: str) -> str:
    return _original_igsm_var(var)


def _igsm_var_for_style(var: str, *, use_v_prefix: bool) -> str:
    return _official_var_name(var) if use_v_prefix else _bare_igsm_var(var)


def _original_igsm_var(var: str) -> str:
    var = var.strip()
    if var.startswith("v_"):
        return var[2:]
    return var


def _replace_igsm_token(expr: str, token: str, value: str) -> str:
    return re.sub(rf"\b{re.escape(token)}\b", str(value), expr)


def _normalize_igsm_expr(expr: str, *, use_v_prefix: bool) -> str:
    normalized = (expr or "").strip()
    if use_v_prefix:
        return re.sub(r"\b([A-Za-z])\b", lambda m: _official_var_name(m.group(1)), normalized)
    return re.sub(r"\bv_([A-Za-z0-9_]+)\b", lambda m: m.group(1), normalized)


def _normalize_igsm_formula(lhs: str, rhs: str) -> str:
    use_v_prefix = lhs.strip().startswith("v_")
    return f"{_igsm_var_for_style(lhs, use_v_prefix=use_v_prefix)} = {_normalize_igsm_expr(rhs, use_v_prefix=use_v_prefix)}"


def _numeric_rhs(formula: str) -> tuple[str, int] | None:
    m = re.match(r"^\s*(v_[A-Za-z0-9_]+|[A-Za-z])\s*=\s*(-?\d+)\s*$", formula)
    if not m:
        return None
    return m.group(1), int(m.group(2)) % 23


@dataclass
class _IgsmTranslationState:
    premise_by_formula: dict[str, int]
    const_by_name: dict[str, str] | None = None
    current_var: str | None = None
    current_expr: str | None = None
    current_line: int | None = None
    known_lines: dict[str, int] | None = None
    known_values: dict[str, int] | None = None
    aliases: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.const_by_name is None:
            self.const_by_name = {}
        if self.known_lines is None:
            self.known_lines = {}
        if self.known_values is None:
            self.known_values = {}
        if self.aliases is None:
            self.aliases = {}

    def remember_if_numeric(self, formula: str, line_number: int) -> None:
        parsed = _numeric_rhs(formula)
        if parsed is None:
            return
        var, value = parsed
        original = _original_igsm_var(var)
        assert self.known_lines is not None and self.known_values is not None
        self.known_lines[original] = line_number
        self.known_values[original] = value

    def canonical_var_for(self, raw_var: str, quantity: str | None = None) -> str:
        assert self.aliases is not None and self.const_by_name is not None
        original = _original_igsm_var(raw_var)
        if quantity:
            for variant in _igsm_quantity_variants(quantity):
                mapped = self.const_by_name.get(_norm(variant))
                if mapped:
                    self.aliases[original] = mapped
                    return mapped
        mapped = self.aliases.get(original)
        if mapped:
            return mapped
        return _igsm_var_for_style(raw_var, use_v_prefix=raw_var.strip().startswith("v_"))

    def canonicalize_expr(self, expr: str, *, use_v_prefix: bool) -> str:
        normalized = _normalize_igsm_expr(expr, use_v_prefix=use_v_prefix)
        assert self.aliases is not None
        for raw, canonical in sorted(self.aliases.items(), key=lambda item: -len(item[0])):
            normalized = _replace_igsm_token(normalized, _official_var_name(raw), canonical)
            normalized = _replace_igsm_token(normalized, _bare_igsm_var(raw), canonical)
        return normalized


_IGSM_RELATION_RE = re.compile(
    r"(?is)^from\s+(?:"
    r"the\s+official\s+igsm\s+relation,\s*"
    r"|(?:the\s+)?(?:igsm\s+)?(?P<relation_kind>definition\s+of|intermediate\s+calculation\s+for)\s+"
    r"(?P<defined_quantity>.+?)\s+\(\s*(?P<defined_lhs>v_[A-Za-z0-9_]+|[A-Za-z])\s*\),\s*"
    r")(?P<lhs>v_[A-Za-z0-9_]+|[A-Za-z])\s+equals\s+(?P<rhs>.+?)\s*$"
)
_IGSM_SUBSTITUTE_RE = re.compile(
    r"(?is)^substitute\s+(?P<var>v_[A-Za-z0-9_]+|[A-Za-z])\s*=\s*(?P<value>-?\d+)\s+into\s+the\s+current\s+expression\s*$"
)
_IGSM_MOD_RE = re.compile(
    r"(?is)^(?:evaluate\s+the\s+arithmetic|reduce\s+the\s+value)\s+modulo\s+23\s+to\s+get\s+"
    r"(?P<lhs>v_[A-Za-z0-9_]+|[A-Za-z])\s*=\s*(?P<value>-?\d+)\s*$"
)


def _translate_igsm_line(text: str, *, state: _IgsmTranslationState, line_number: int) -> tuple[str, str] | None:
    clause = _unwrap_controlled_trace_line(text).strip().rstrip(".").strip()
    relation = _IGSM_RELATION_RE.match(clause)
    if relation:
        lhs = state.canonical_var_for(
            relation.group("lhs"),
            quantity=(
                relation.groupdict().get("defined_quantity")
                if (relation.groupdict().get("relation_kind") or "").lower().strip() == "definition of"
                else None
            ),
        )
        rhs = state.canonicalize_expr(relation.group("rhs"), use_v_prefix=lhs.startswith("v_"))
        formula = f"{lhs} = {rhs}"
        premise_line = state.premise_by_formula.get(_canonical_formula_text(formula), 1)
        state.current_var = formula.split("=", 1)[0].strip()
        state.current_expr = formula.split("=", 1)[1].strip()
        state.current_line = line_number
        state.remember_if_numeric(formula, line_number)
        assert state.aliases is not None and state.known_lines is not None
        raw_original = _original_igsm_var(relation.group("lhs"))
        state.aliases[raw_original] = state.current_var
        state.known_lines[raw_original] = line_number
        return formula, f"R,{premise_line}"

    substitute = _IGSM_SUBSTITUTE_RE.match(clause)
    if substitute:
        if not state.current_var or state.current_expr is None or state.current_line is None:
            return None
        raw_var = substitute.group("var")
        value = str(int(substitute.group("value")) % 23)
        original = _original_igsm_var(raw_var)
        canonical = state.canonical_var_for(raw_var)
        state.current_expr = _replace_igsm_token(state.current_expr, _official_var_name(raw_var), value)
        state.current_expr = _replace_igsm_token(state.current_expr, _bare_igsm_var(raw_var), value)
        state.current_expr = _replace_igsm_token(state.current_expr, canonical, value)
        formula = f"{state.current_var} = {state.current_expr}"
        known_line = (state.known_lines or {}).get(original, (state.known_lines or {}).get(_original_igsm_var(canonical), 1))
        justification = f"=E,{known_line},{state.current_line}"
        state.current_line = line_number
        state.remember_if_numeric(formula, line_number)
        return formula, justification

    mod_line = _IGSM_MOD_RE.match(clause)
    if mod_line:
        lhs = state.canonical_var_for(mod_line.group("lhs"))
        formula = f"{lhs} = {int(mod_line.group('value')) % 23}"
        cite = state.current_line if state.current_line is not None else 1
        state.current_var = lhs
        state.current_expr = str(int(mod_line.group("value")) % 23)
        state.current_line = line_number
        state.remember_if_numeric(formula, line_number)
        return formula, f"MOD23,{cite}"

    return None


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
    constants: str | Iterable[str] | None,
    predicates: str | Iterable[str] | None,
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
        formulas.append(_predicate_atom(pred, entity))
    if not formulas:
        return None
    return " & ".join(formulas)


def translate_natural_proof_to_fol(
    proof_text: str,
    *,
    constants: str | Iterable[str] | None,
    predicates: str | Iterable[str] | None,
    premise_count: int,
    premises: str | Iterable[str] | None = None,
) -> NaturalProofTranslation:
    proof_lines: list[str] = []
    errors: list[str] = []
    parsed = 0
    total = 0
    next_line = int(premise_count) + 1
    igsm_state = _IgsmTranslationState(
        premise_by_formula=_proof_premise_map(premises),
        const_by_name=_constant_map(constants),
    )
    for raw in (proof_text or "").splitlines():
        text = raw.strip()
        if not text:
            continue
        total += 1
        translated_igsm = _translate_igsm_line(text, state=igsm_state, line_number=next_line)
        if translated_igsm is not None:
            formula, justification = translated_igsm
        else:
            formula = translate_natural_sentence_to_formula(text, constants=constants, predicates=predicates)
            justification = "R"
        if formula is None:
            formula = "INVALID"
            errors.append(f"line {total}: could not translate `{_strip_number(text)}`")
        else:
            parsed += 1
        proof_lines.append(f"{next_line}. {formula} ; {justification}")
        next_line += 1
    return NaturalProofTranslation(
        proof="\n".join(proof_lines),
        parsed_lines=parsed,
        total_lines=total,
        errors=tuple(errors),
    )
