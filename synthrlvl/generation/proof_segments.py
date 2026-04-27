from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FormalProofDraft:
    generation_prefix_through_proof: str
    prompt_prefix_through_proof: str
    premises: str
    available: bool


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def draft_from_generation(prompt: str, generation: str) -> FormalProofDraft:
    """Extract the model-generated formal prefix up to an open <proof> tag."""
    text = generation or ""
    proof_match = re.search(r"<proof>\s*", text, flags=re.IGNORECASE)
    if not proof_match:
        return FormalProofDraft(generation_prefix_through_proof=text, prompt_prefix_through_proof=prompt + text, premises="", available=False)

    prefix = text[: proof_match.end()]
    formal_before_proof = text[: proof_match.start()]
    premises = extract_tag(formal_before_proof, "premises")
    if not premises:
        return FormalProofDraft(generation_prefix_through_proof=prefix, prompt_prefix_through_proof=prompt + prefix, premises="", available=False)
    return FormalProofDraft(generation_prefix_through_proof=prefix, prompt_prefix_through_proof=prompt + prefix, premises=premises, available=True)


def compose_final_output(*, prefix_through_proof: str, proof_lines: list[str], suffix: str) -> str:
    proof = "\n".join(line for line in proof_lines if line.strip())
    body = prefix_through_proof
    if proof:
        body += proof + "\n"
    if "</proof>" not in suffix.lower():
        body += "</proof>\n"
    body += suffix.lstrip()
    return body
