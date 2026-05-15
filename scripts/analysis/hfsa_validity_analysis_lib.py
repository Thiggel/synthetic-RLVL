from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from synthrlvl.metrics import OutputEvaluator, extract_tag, _is_answer_match
from synthrlvl.types import PrefillMode, TemplateName

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ATOM_RE = re.compile(r"^\s*([A-Z])([a-z])\s*$")
PRED_RE = re.compile(r"^\s*([A-Z])x\s*:\s*x\s+is\s+(.+?)\s*$", re.IGNORECASE)
QUERY_RE = re.compile(r"Which state applies to\s+([a-z])\?", re.IGNORECASE)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def split_nonempty(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def strip_line_number(line: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", line or "").strip()


def parse_predicate_map(generation: str) -> dict[str, str]:
    formal = extract_tag(generation, "formal") or generation or ""
    preds = extract_tag(formal, "predicates")
    out: dict[str, str] = {}
    for raw in split_nonempty(preds):
        line = strip_line_number(raw)
        m = PRED_RE.match(line)
        if m:
            out[m.group(1).upper()] = m.group(2).strip().lower()
    return out


def parse_atom(formula: str) -> tuple[str, str] | None:
    m = ATOM_RE.match((formula or "").strip())
    if not m:
        return None
    return m.group(1).upper(), m.group(2).lower()


def proof_formulas(generation: str) -> list[str]:
    formal = extract_tag(generation, "formal") or generation or ""
    proof = extract_tag(formal, "proof")
    formulas: list[str] = []
    for raw in split_nonempty(proof):
        line = strip_line_number(raw)
        formula = line.split(";", 1)[0].strip()
        if formula:
            formulas.append(formula)
    return formulas


def premise_formulas(generation: str) -> list[str]:
    formal = extract_tag(generation, "formal") or generation or ""
    premises = extract_tag(formal, "premises")
    return [strip_line_number(raw) for raw in split_nonempty(premises)]


def conclusion_formula(generation: str) -> str:
    formal = extract_tag(generation, "formal") or generation or ""
    return strip_line_number(extract_tag(formal, "conclusion"))


def query_constant_from_prompt(prompt: str, metadata: dict[str, Any] | None = None) -> str:
    if metadata and metadata.get("query_constant"):
        return str(metadata["query_constant"]).strip().lower()
    matches = QUERY_RE.findall(prompt or "")
    return matches[-1].lower() if matches else ""


def words_to_predicates(pred_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for pred, word in pred_map.items():
        out.setdefault(norm_text(word), []).append(pred)
    return out


def _word_match(a: str, b: str) -> bool:
    return _is_answer_match(a or "", b or "")


def classify_generation(
    *,
    generation: str,
    prompt: str,
    step: int,
    gold_answer: str,
    shortcut_answer: str | None,
    metadata: dict[str, Any] | None = None,
    source: str = "targeted",
) -> dict[str, Any]:
    evaluator = OutputEvaluator()
    score = evaluator.evaluate(
        generation,
        template=TemplateName.LOGIC,
        gold_answer=gold_answer,
        gold_logic_premises="",
        gold_logic_conclusion="",
        prefill=PrefillMode.NONE,
        gold_first_modality_lines=[],
    )
    formal = extract_tag(generation, "formal") or generation or ""
    premises, proof, conclusion = evaluator._extract_logic_components(generation, "formal")
    citation_free_line_valid_fraction = 0.0
    line_valid_fraction = 0.0
    citation_free_error = ""
    strict_error = ""
    if premises and proof and conclusion:
        try:
            strict_report = evaluator.engine.analyze_proof(premises=premises, conclusion=conclusion, proof=proof)
            if strict_report.lines:
                line_valid_fraction = sum(1 for line in strict_report.lines if line.valid) / len(strict_report.lines)
            strict_error = strict_report.error or ""
        except Exception as exc:  # pragma: no cover - diagnostic only
            strict_error = repr(exc)
        try:
            cf_report = evaluator.engine.analyze_proof_citation_free(premises=premises, conclusion=conclusion, proof=proof)
            if cf_report.lines:
                citation_free_line_valid_fraction = sum(1 for line in cf_report.lines if line.valid) / len(cf_report.lines)
            citation_free_error = cf_report.error or ""
        except Exception as exc:  # pragma: no cover - diagnostic only
            citation_free_error = repr(exc)

    answer = extract_tag(generation, "answer")
    pred_map = parse_predicate_map(generation)
    word_to_preds = words_to_predicates(pred_map)
    query_const = query_constant_from_prompt(prompt, metadata)
    formulas = proof_formulas(generation)
    atoms = [(formula, parse_atom(formula)) for formula in formulas]
    atom_texts = [
        {
            "formula": formula,
            "pred": atom[0],
            "const": atom[1],
            "text": pred_map.get(atom[0], ""),
        }
        for formula, atom in atoms
        if atom is not None
    ]
    answer_norm = norm_text(answer)
    gold_norm = norm_text(gold_answer)
    shortcut_norm = norm_text(shortcut_answer or "")
    answer_preds = set(word_to_preds.get(answer_norm, []))
    gold_preds = set(word_to_preds.get(gold_norm, []))
    shortcut_preds = set(word_to_preds.get(shortcut_norm, []))

    def has_atom(preds: set[str], const: str | None = None) -> bool:
        for item in atom_texts:
            if item["pred"] in preds and (not const or item["const"] == const):
                return True
        return False

    conc = conclusion_formula(generation)
    conc_atom = parse_atom(conc)
    conc_pred = conc_atom[0] if conc_atom else ""
    conc_const = conc_atom[1] if conc_atom else ""
    conc_text = pred_map.get(conc_pred, "") if conc_pred else ""

    proof_line_count = len(formulas)
    expected_proof_lines = int((metadata or {}).get("expected_proof_lines", 2 * int(step) + 2))
    answer_is_gold = _word_match(answer, gold_answer)
    answer_is_shortcut = bool(shortcut_answer) and _word_match(answer, str(shortcut_answer))
    conclusion_is_answer = bool(conc_text) and _word_match(conc_text, answer)
    conclusion_is_gold = bool(conc_text) and _word_match(conc_text, gold_answer)
    conclusion_is_shortcut = bool(shortcut_answer and conc_text) and _word_match(conc_text, str(shortcut_answer))
    conclusion_const_is_query = bool(query_const and conc_const == query_const)
    answer_atom_derived_query = has_atom(answer_preds, query_const) if answer_preds and query_const else False
    gold_atom_derived_query = has_atom(gold_preds, query_const) if gold_preds and query_const else False
    shortcut_atom_derived_query = has_atom(shortcut_preds, query_const) if shortcut_preds and query_const else False
    answer_derived_by_conclusion = bool(score.citation_free_valid and conclusion_is_answer and conclusion_const_is_query)
    correct_answer_derived = bool(answer_is_gold and answer_derived_by_conclusion)
    correct_gold_atom_derived_query = bool(answer_is_gold and gold_atom_derived_query)
    correct_cfvalid_gold_atom_derived = bool(answer_is_gold and score.citation_free_valid > 0 and gold_atom_derived_query)

    if score.format_ok <= 0:
        category = "malformed"
    elif correct_cfvalid_gold_atom_derived:
        category = "correct_cfvalid_gold_atom_derived"
    elif answer_is_gold and score.citation_free_valid > 0:
        category = "correct_cfvalid_no_gold_atom"
    elif correct_gold_atom_derived_query:
        category = "correct_cfinvalid_gold_atom_derived"
    elif answer_is_gold:
        category = "correct_cfinvalid_no_gold_atom"
    elif answer_is_shortcut:
        category = "shortcut_wrong"
    elif score.citation_free_valid > 0 and gold_atom_derived_query:
        category = "wrong_cfvalid_gold_atom_derived"
    elif score.citation_free_valid > 0:
        category = "wrong_cfvalid_no_gold_atom"
    else:
        category = "wrong_cfinvalid"

    return {
        "source": source,
        "step": int(step),
        "query_constant": query_const,
        "answer_text": answer,
        "gold_answer": gold_answer,
        "shortcut_answer": shortcut_answer,
        "answer_is_gold": float(answer_is_gold),
        "answer_is_shortcut": float(answer_is_shortcut),
        "syntactic": float(score.syntactic),
        "format_ok": float(score.format_ok),
        "correct": float(score.correct),
        "valid": float(score.valid),
        "citation_free_valid": float(score.citation_free_valid),
        "line_valid_fraction": float(line_valid_fraction),
        "citation_free_line_valid_fraction": float(citation_free_line_valid_fraction),
        "strict_validity_error": strict_error,
        "citation_free_validity_error": citation_free_error,
        "predicate_count": len(pred_map),
        "premise_line_count": len(premise_formulas(generation)),
        "proof_line_count": proof_line_count,
        "expected_proof_lines": expected_proof_lines,
        "proof_line_ratio": proof_line_count / expected_proof_lines if expected_proof_lines else 0.0,
        "proof_shorter_than_expected": float(proof_line_count < expected_proof_lines),
        "proof_longer_than_expected": float(proof_line_count > expected_proof_lines),
        "conclusion_formula": conc,
        "conclusion_predicate": conc_pred,
        "conclusion_constant": conc_const,
        "conclusion_text": conc_text,
        "conclusion_is_answer": float(conclusion_is_answer),
        "conclusion_is_gold": float(conclusion_is_gold),
        "conclusion_is_shortcut": float(conclusion_is_shortcut),
        "conclusion_const_is_query": float(conclusion_const_is_query),
        "answer_atom_derived_query": float(answer_atom_derived_query),
        "gold_atom_derived_query": float(gold_atom_derived_query),
        "shortcut_atom_derived_query": float(shortcut_atom_derived_query),
        "answer_derived_by_conclusion": float(answer_derived_by_conclusion),
        "correct_answer_derived": float(correct_answer_derived),
        "correct_gold_atom_derived_query": float(correct_gold_atom_derived_query),
        "correct_cfvalid_gold_atom_derived": float(correct_cfvalid_gold_atom_derived),
        "category": category,
        "proof_formulas": formulas,
        "atom_texts": atom_texts,
    }


def parse_run_name(name: str) -> dict[str, Any]:
    base = name.replace("_passk.json", "").replace("_samples.jsonl", "")
    m = re.search(r"rl_hfsa_easy500_train(?P<shortcut>0p\d+)_(?P<schema>.+)_seed(?P<seed>\d+)_mrg", base)
    if not m:
        return {"run_name": base, "shortcut_tag": "unknown", "schema": "unknown", "seed": -1, "condition": base}
    shortcut = m.group("shortcut")
    schema = m.group("schema")
    seed = int(m.group("seed"))
    reward_short = "validity_gated" if "citation_free_valid" in schema else "correct_only"
    return {
        "run_name": base,
        "shortcut_tag": shortcut,
        "schema": schema,
        "seed": seed,
        "reward_short": reward_short,
        "condition": f"train{shortcut}_{reward_short}",
    }


def iter_metric_pairs_from_log_line(line: str) -> Iterable[tuple[str, float]]:
    clean = strip_ansi(line)
    for key, raw in re.findall(r"([A-Za-z0-9_/]+(?:/[A-Za-z0-9_]+)*):([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", clean):
        try:
            yield key, float(raw)
        except ValueError:
            continue
