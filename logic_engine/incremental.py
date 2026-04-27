from __future__ import annotations

from dataclasses import dataclass

from .api import LogicEngine


@dataclass(frozen=True)
class CandidateLineReport:
    line: str
    syntactic: bool
    valid: bool
    novel: bool
    score: int
    error: str | None = None


@dataclass(frozen=True)
class ProofState:
    premises: str
    accepted_lines: tuple[str, ...] = ()

    @property
    def proof_text(self) -> str:
        return "\n".join(self.accepted_lines)


class IncrementalProofValidator:
    """Line-level proof validator used by constrained decoding.

    This intentionally delegates to the existing full ProofAnalyzer so the
    incremental decoder stays behaviorally aligned with normal validity
    metrics. The API is incremental; the implementation can later be optimized
    without changing generation code.
    """

    def __init__(self, engine: LogicEngine | None = None):
        self.engine = engine or LogicEngine()

    def initial_state(self, premises: str) -> ProofState:
        return ProofState(premises=(premises or "").strip(), accepted_lines=())

    def check_next_line(self, state: ProofState, line: str) -> CandidateLineReport:
        normalized = normalize_candidate_line(line)
        if not normalized:
            return CandidateLineReport(line="", syntactic=False, valid=False, novel=False, score=0, error="empty line")
        proof = "\n".join((*state.accepted_lines, normalized))
        try:
            report = self.engine.analyze_proof(premises=state.premises, conclusion=None, proof=proof)
        except Exception as exc:
            return CandidateLineReport(line=normalized, syntactic=False, valid=False, novel=False, score=0, error=str(exc))
        if not report.lines:
            return CandidateLineReport(line=normalized, syntactic=False, valid=False, novel=False, score=0, error=report.error)
        last = report.lines[-1]
        syntactic = bool(last.syntax_valid)
        valid = bool(last.valid)
        novel = bool(last.novel)
        score = candidate_score(syntactic=syntactic, valid=valid, novel=novel)
        return CandidateLineReport(
            line=normalized,
            syntactic=syntactic,
            valid=valid,
            novel=novel,
            score=score,
            error=last.error or last.syntax_error or report.error,
        )

    def accept_line(self, state: ProofState, line: str) -> ProofState:
        normalized = normalize_candidate_line(line)
        return ProofState(premises=state.premises, accepted_lines=(*state.accepted_lines, normalized))


def candidate_score(*, syntactic: bool, valid: bool, novel: bool) -> int:
    if valid and novel:
        return 3
    if valid:
        return 2
    if syntactic:
        return 1
    return 0


def normalize_candidate_line(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    first = stripped.splitlines()[0].strip()
    # vLLM may include the closing proof tag because it is not token-level constrained.
    first = first.split("</proof>", 1)[0].strip()
    first = first.split("<conclusion>", 1)[0].strip()
    if first.lower().startswith("next line:"):
        first = first.split(":", 1)[1].strip()
    if first.startswith("- "):
        first = first[2:].strip()
    return first
