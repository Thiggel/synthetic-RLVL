from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from .mathkernels import MathKernels
from .parser import ParsingError, parse_assumption, parse_formula, parse_line
from .prover import (
    And,
    Eq,
    Exists,
    FOL,
    Falsum,
    Forall,
    Iff,
    Imp,
    Justification,
    Not,
    Or,
    Pred,
    Proof,
    PropVar,
    is_fol_sentence,
)


@dataclass(frozen=True)
class PremiseReport:
    index: int
    line_number: int
    text: str
    syntax_valid: bool
    relevant: bool = False
    novel: bool = False
    relevant_and_novel: bool = False
    fact_key: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LineReport:
    index: int
    line_number: int
    text: str
    syntax_valid: bool
    valid: bool = False
    relevant: bool = False
    novel: bool = False
    relevant_and_novel: bool = False
    fact_key: str | None = None
    dependencies: tuple[int, ...] = ()
    syntax_error: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProofGraph:
    # nodes are source line numbers (premise + proof lines)
    nodes: tuple[int, ...]
    # edges are (from_dependency, to_dependent)
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ProofReport:
    ok: bool
    rendered: str
    error: str | None
    conclusion_supported: bool
    premises: tuple[PremiseReport, ...]
    lines: tuple[LineReport, ...]
    graph: ProofGraph


class FormulaEquivalence:
    @staticmethod
    def equivalent(formula_a: Any, formula_b: Any) -> bool:
        if formula_a == formula_b:
            return True
        if isinstance(formula_a, Eq) and isinstance(formula_b, Eq):
            return MathKernels.equations_equivalent(formula_a, formula_b)
        return FormulaEquivalence._canonical_formula(formula_a) == FormulaEquivalence._canonical_formula(formula_b)

    @staticmethod
    def equivalent_text(formula_a: str, formula_b: str) -> bool:
        parsed_a = parse_formula(formula_a)
        parsed_b = parse_formula(formula_b)
        return FormulaEquivalence.equivalent(parsed_a, parsed_b)

    @staticmethod
    def _canonical_formula(formula: Any, env: dict[str, str] | None = None, bound_idx: int = 0) -> str:
        env = {} if env is None else dict(env)

        if isinstance(formula, Imp):
            return FormulaEquivalence._canonical_formula(Or(Not(formula.left), formula.right), env, bound_idx)
        if isinstance(formula, Iff):
            left = FormulaEquivalence._canonical_formula(Imp(formula.left, formula.right), env, bound_idx)
            right = FormulaEquivalence._canonical_formula(Imp(formula.right, formula.left), env, bound_idx)
            pair = sorted((left, right))
            return f"(iff {pair[0]} {pair[1]})"
        if isinstance(formula, Not):
            inner = formula.inner
            if isinstance(inner, Not):
                return FormulaEquivalence._canonical_formula(inner.inner, env, bound_idx)
            if isinstance(inner, And):
                return FormulaEquivalence._canonical_formula(Or(Not(inner.left), Not(inner.right)), env, bound_idx)
            if isinstance(inner, Or):
                return FormulaEquivalence._canonical_formula(And(Not(inner.left), Not(inner.right)), env, bound_idx)
            if isinstance(inner, Imp):
                return FormulaEquivalence._canonical_formula(And(inner.left, Not(inner.right)), env, bound_idx)
            if isinstance(inner, Forall):
                return FormulaEquivalence._canonical_formula(Exists(inner.var, Not(inner.inner)), env, bound_idx)
            if isinstance(inner, Exists):
                return FormulaEquivalence._canonical_formula(Forall(inner.var, Not(inner.inner)), env, bound_idx)
            return f"(not {FormulaEquivalence._canonical_formula(inner, env, bound_idx)})"
        if isinstance(formula, And):
            parts = FormulaEquivalence._flatten(formula, And)
            keys = sorted(FormulaEquivalence._canonical_formula(p, env, bound_idx) for p in parts)
            return f"(and {' '.join(keys)})"
        if isinstance(formula, Or):
            parts = FormulaEquivalence._flatten(formula, Or)
            keys = sorted(FormulaEquivalence._canonical_formula(p, env, bound_idx) for p in parts)
            return f"(or {' '.join(keys)})"
        if isinstance(formula, Forall):
            local = f"v{bound_idx}"
            env[formula.var.name] = local
            body = FormulaEquivalence._canonical_formula(formula.inner, env, bound_idx + 1)
            return f"(forall {local} {body})"
        if isinstance(formula, Exists):
            local = f"v{bound_idx}"
            env[formula.var.name] = local
            body = FormulaEquivalence._canonical_formula(formula.inner, env, bound_idx + 1)
            return f"(exists {local} {body})"
        if isinstance(formula, Eq):
            left = FormulaEquivalence._canonical_term(formula.left, env)
            right = FormulaEquivalence._canonical_term(formula.right, env)
            ordered = sorted((left, right))
            return f"(eq {ordered[0]} {ordered[1]})"
        if isinstance(formula, Pred):
            args = " ".join(FormulaEquivalence._canonical_term(arg, env) for arg in formula.args)
            return f"(pred {formula.name} {args})".strip()
        if isinstance(formula, PropVar):
            return f"(prop {formula.name})"
        if isinstance(formula, Falsum):
            return "(false)"
        return str(formula)

    @staticmethod
    def _canonical_term(term: Any, env: dict[str, str]) -> str:
        if hasattr(term, "fname") and hasattr(term, "args"):
            args = " ".join(FormulaEquivalence._canonical_term(a, env) for a in term.args)
            return f"({term.fname} {args})".strip()
        name = getattr(term, "name", str(term))
        return env.get(name, name)

    @staticmethod
    def _flatten(formula: Any, typ: type) -> list[Any]:
        if isinstance(formula, typ):
            return FormulaEquivalence._flatten(formula.left, typ) + FormulaEquivalence._flatten(formula.right, typ)
        return [formula]


class ProofAnalyzer:
    def analyze(self, premises: str, conclusion: str | None, proof_text: str) -> ProofReport:
        premise_lines = self._split_premise_lines(premises)
        parsed_premises: list[Any] = []
        analyzed_premises: list[PremiseReport] = []
        source_to_internal: dict[int, int] = {}
        internal_to_source: dict[int, int] = {}
        first_error: str | None = None

        for premise in premise_lines:
            try:
                parsed = self._parse_fol_sentence(premise.text)
                if isinstance(parsed, Falsum):
                    raise ParsingError('premise "#" is not allowed')
                parsed_premises.append(parsed)
                internal_idx = len(parsed_premises)
                source_to_internal[premise.line_number] = internal_idx
                internal_to_source[internal_idx] = premise.line_number
                analyzed_premises.append(
                    replace(premise, syntax_valid=True, fact_key=self._fact_key(parsed))
                )
            except Exception as exc:
                if first_error is None:
                    first_error = f"premise parse failed: {exc}"
                analyzed_premises.append(replace(premise, syntax_valid=False, error=str(exc)))

        conclusion_text = (conclusion or self._infer_conclusion(proof_text) or "").strip()
        proof_lines = self._split_proof_lines(proof_text, premise_count=len(analyzed_premises))
        annotated: list[LineReport] = []
        for line in proof_lines:
            syntax_error = self._line_syntax_error(line.text)
            if syntax_error is None:
                annotated.append(replace(line, syntax_valid=True))
            else:
                annotated.append(replace(line, syntax_valid=False, syntax_error=syntax_error))

        if not conclusion_text:
            return self._empty_report(analyzed_premises, annotated, first_error or "No conclusion found.")

        try:
            parsed_conclusion = self._parse_fol_sentence(conclusion_text)
            proof = Proof(FOL, parsed_premises, parsed_conclusion)
        except Exception as exc:
            return self._empty_report(analyzed_premises, annotated, first_error or f"proof init failed: {exc}")

        valid_lines: list[LineReport] = []
        deps_by_internal: dict[int, tuple[int, ...]] = {}
        conclusion_candidates: list[int] = []

        for line in annotated:
            if not line.syntax_valid:
                if first_error is None:
                    first_error = line.syntax_error
                valid_lines.append(replace(line, valid=False, error=line.syntax_error))
                continue
            try:
                formula, remapped = self._apply_proof_line(
                    proof,
                    line.text,
                    source_to_internal=source_to_internal,
                    source_line_number=line.line_number,
                )
                internal = proof.proof.idx[1]
                source_to_internal[line.line_number] = internal
                internal_to_source[internal] = line.line_number
                deps_by_internal[internal] = remapped
                if formula == parsed_conclusion:
                    conclusion_candidates.append(internal)
                fact_key = None if isinstance(formula, Falsum) else self._fact_key(formula)
                valid_lines.append(
                    replace(line, valid=True, fact_key=fact_key, dependencies=tuple(internal_to_source[d] for d in remapped if d in internal_to_source))
                )
            except Exception as exc:
                if first_error is None:
                    first_error = str(exc)
                valid_lines.append(replace(line, valid=False, error=str(exc)))

        relevant_premise_indexes, relevant_line_indexes = self._trace_relevance(
            premise_count=len(parsed_premises),
            deps_by_internal=deps_by_internal,
            internal_to_line_index={internal: i for i, (internal, _) in enumerate(sorted(internal_to_source.items()))},
            conclusion_candidates=conclusion_candidates,
            source_line_for_internal=internal_to_source,
            analyzed_lines=valid_lines,
        )

        analyzed_premises = [
            replace(p, relevant=(p.index in relevant_premise_indexes)) if p.syntax_valid else p
            for p in analyzed_premises
        ]
        valid_lines = [
            replace(l, relevant=(l.index in relevant_line_indexes)) if l.valid else l
            for l in valid_lines
        ]

        seen_keys: set[str] = set()
        analyzed_premises = [
            replace(
                premise,
                novel=(premise.fact_key is not None and premise.fact_key not in seen_keys and not seen_keys.add(premise.fact_key)),
            )
            if premise.syntax_valid
            else premise
            for premise in analyzed_premises
        ]
        for idx, premise in enumerate(analyzed_premises):
            if premise.syntax_valid:
                analyzed_premises[idx] = replace(
                    premise, relevant_and_novel=premise.relevant and premise.novel
                )

        for idx, line in enumerate(valid_lines):
            if line.valid and line.fact_key is not None:
                is_novel = line.fact_key not in seen_keys
                if is_novel:
                    seen_keys.add(line.fact_key)
                valid_lines[idx] = replace(
                    line,
                    novel=is_novel,
                    relevant_and_novel=line.relevant and is_novel,
                )

        edges = tuple(
            (dep, line.line_number)
            for line in valid_lines
            for dep in line.dependencies
        )
        nodes = tuple(sorted({p.line_number for p in analyzed_premises} | {l.line_number for l in valid_lines}))
        graph = ProofGraph(nodes=nodes, edges=edges)

        invalid_count = sum(1 for line in valid_lines if not line.valid)
        return ProofReport(
            ok=(invalid_count == 0 and proof.is_complete()),
            rendered=str(proof),
            error=first_error,
            conclusion_supported=bool(conclusion_candidates),
            premises=tuple(analyzed_premises),
            lines=tuple(valid_lines),
            graph=graph,
        )

    def _empty_report(self, premises: list[PremiseReport], lines: list[LineReport], error: str) -> ProofReport:
        nodes = tuple(sorted({p.line_number for p in premises} | {l.line_number for l in lines}))
        return ProofReport(
            ok=False,
            rendered="",
            error=error,
            conclusion_supported=False,
            premises=tuple(premises),
            lines=tuple(lines),
            graph=ProofGraph(nodes=nodes, edges=()),
        )

    @staticmethod
    def _parse_fol_sentence(text: str):
        formula = parse_formula(text)
        if not is_fol_sentence(formula):
            raise ParsingError(f'"{formula}" is not a valid FOL sentence.')
        return formula

    @staticmethod
    def _infer_conclusion(proof_text: str) -> str | None:
        lines = [line.strip() for line in proof_text.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            return None
        last_line = re.sub(r"^\s*\d+\.?\s*", "", lines[-1])
        if last_line.lower().startswith("end:"):
            last_line = last_line[4:].strip()
        return last_line.split(";", 1)[0].strip() or None

    @staticmethod
    def _split_premise_lines(text: str) -> list[PremiseReport]:
        raw = (text or "").strip()
        if not raw or raw.upper() == "NA":
            return []
        base_parts = [p.strip() for p in re.split(r"[;\n]", raw) if p.strip()]
        if len(base_parts) == 1 and "," in raw:
            base_parts = [p.strip() for p in raw.split(",") if p.strip()]
        parts = [re.sub(r"^\s*\d+\.?\s*", "", part).strip() for part in base_parts if part.strip()]
        return [
            PremiseReport(index=i, line_number=i + 1, text=part, syntax_valid=False)
            for i, part in enumerate(parts)
        ]

    @staticmethod
    def _split_proof_lines(text: str, premise_count: int) -> list[LineReport]:
        lines: list[LineReport] = []
        next_num = premise_count + 1
        for idx, raw in enumerate(text.splitlines()):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = re.match(r"^\s*(\d+)\.\s*(.*)$", stripped, flags=re.DOTALL)
            if m:
                line_number = int(m.group(1))
                line_text = m.group(2).strip()
            else:
                line_number = next_num
                line_text = stripped
            next_num = max(next_num + 1, line_number + 1)
            lines.append(
                LineReport(index=len(lines), line_number=line_number, text=line_text, syntax_valid=False)
            )
        return lines

    @staticmethod
    def _line_syntax_error(line_text: str) -> str | None:
        try:
            if line_text.lower().startswith("assume:"):
                parse_assumption(line_text[len("assume:") :].strip())
            elif line_text.lower().startswith("endbegin:"):
                parse_assumption(line_text[len("endbegin:") :].strip())
            elif line_text.lower().startswith("end:"):
                parse_line(line_text[len("end:") :].strip())
            else:
                parse_line(line_text)
            return None
        except Exception as exc:
            return str(exc)

    @staticmethod
    def _remap_citations(citations: tuple, source_to_internal: dict[int, int], current_source_line: int) -> tuple:
        remapped: list[int | tuple[int, int]] = []
        for citation in citations:
            if isinstance(citation, int):
                if citation >= current_source_line:
                    raise ParsingError(f"Citation {citation} references a future line.")
                if citation not in source_to_internal:
                    raise ParsingError(f"Citation {citation} references an unavailable line.")
                remapped.append(source_to_internal[citation])
            else:
                start, end = citation
                if start >= current_source_line or end >= current_source_line:
                    raise ParsingError(f"Citation {start}-{end} references a future line.")
                expanded: list[int] = []
                for n in range(start, end + 1):
                    if n not in source_to_internal:
                        raise ParsingError(f"Citation {start}-{end} references an unavailable line {n}.")
                    expanded.append(source_to_internal[n])
                if expanded != list(range(expanded[0], expanded[-1] + 1)):
                    raise ParsingError(f"Citation {start}-{end} is not contiguous after invalid lines were removed.")
                remapped.append((expanded[0], expanded[-1]))
        return tuple(remapped)

    @staticmethod
    def _flatten_citations(citations: tuple) -> tuple[int, ...]:
        out: list[int] = []
        for citation in citations:
            if isinstance(citation, int):
                out.append(citation)
            else:
                start, end = citation
                out.extend(range(start, end + 1))
        return tuple(out)

    def _apply_proof_line(
        self,
        proof: Proof,
        line_text: str,
        *,
        source_to_internal: dict[int, int],
        source_line_number: int,
    ) -> tuple[Any, tuple[int, ...]]:
        if line_text.lower().startswith("assume:"):
            assumption = parse_assumption(line_text[len("assume:") :].strip())
            proof.begin_subproof(assumption)
            return assumption, ()
        if line_text.lower().startswith("endbegin:"):
            assumption = parse_assumption(line_text[len("endbegin:") :].strip())
            proof.end_and_begin_subproof(assumption)
            return assumption, ()
        if line_text.lower().startswith("end:"):
            formula, justification = parse_line(line_text[len("end:") :].strip())
            remapped = self._remap_citations(justification.citations, source_to_internal, source_line_number)
            proof.end_subproof(formula, Justification(justification.rule, remapped))
            return formula, self._flatten_citations(remapped)
        formula, justification = parse_line(line_text)
        remapped = self._remap_citations(justification.citations, source_to_internal, source_line_number)
        proof.add_line(formula, Justification(justification.rule, remapped))
        return formula, self._flatten_citations(remapped)

    def _trace_relevance(
        self,
        *,
        premise_count: int,
        deps_by_internal: dict[int, tuple[int, ...]],
        internal_to_line_index: dict[int, int],
        conclusion_candidates: list[int],
        source_line_for_internal: dict[int, int],
        analyzed_lines: list[LineReport],
    ) -> tuple[set[int], set[int]]:
        if not conclusion_candidates:
            return set(), set()

        relevant_internal: set[int] = set()
        stack = [conclusion_candidates[-1]]
        while stack:
            node = stack.pop()
            if node in relevant_internal:
                continue
            relevant_internal.add(node)
            stack.extend(deps_by_internal.get(node, ()))

        premise_indexes = {i - 1 for i in relevant_internal if 1 <= i <= premise_count}

        line_by_source = {line.line_number: line for line in analyzed_lines}
        relevant_lines: set[int] = set()
        for internal in relevant_internal:
            source_line = source_line_for_internal.get(internal)
            if source_line is None:
                continue
            line = line_by_source.get(source_line)
            if line is not None:
                relevant_lines.add(line.index)
        return premise_indexes, relevant_lines

    def _fact_key(self, formula: Any) -> str:
        return FormulaEquivalence._canonical_formula(formula)
