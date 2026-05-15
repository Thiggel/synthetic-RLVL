from __future__ import annotations

from dataclasses import dataclass, field

from .analysis import FormulaEquivalence, ProofAnalyzer, ProofReport


@dataclass
class LogicEngine:
    analyzer: ProofAnalyzer = field(default_factory=ProofAnalyzer)

    def validate_proof(self, premises: str, conclusion: str, proof: str) -> bool:
        return self.analyze_proof(premises, conclusion, proof).ok

    def analyze_proof(self, premises: str, conclusion: str | None, proof: str) -> ProofReport:
        return self.analyzer.analyze(premises=premises, conclusion=conclusion, proof_text=proof)

    def validate_proof_citation_free(self, premises: str, conclusion: str, proof: str) -> bool:
        return self.analyze_proof_citation_free(premises, conclusion, proof).ok

    def analyze_proof_citation_free(self, premises: str, conclusion: str | None, proof: str) -> ProofReport:
        return self.analyzer.analyze_citation_free(premises=premises, conclusion=conclusion, proof_text=proof)

    def are_equivalent(self, formula_a: str, formula_b: str) -> bool:
        return FormulaEquivalence.equivalent_text(formula_a, formula_b)
