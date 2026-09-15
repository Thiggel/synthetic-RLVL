"""A checker for a logic block that has to carry the answer.

The reward in `rl_checker_grpo.py` can pay for a derivation that a program
accepts. A derivation is easy to fake unless the program demands three things
at once, so this checker demands all three.

  grounding      every premise quotes a span that occurs verbatim in the
                 prompt, which stops the model from inventing the facts it
                 needs, the failure we observed on multi-hop question
                 answering where the model declared a predicate legend over
                 Wikipedia entities.
  derivation     every step cites earlier lines and a rule from a fixed set,
                 and arithmetic steps are evaluated exactly.
  load-bearing   the answer is read off the final line of the block and has to
                 equal the answer the model states in its own words. A block
                 that derives nothing cannot satisfy this, and a block that
                 simply asserts the answer as a premise is rejected by the
                 non-triviality rule below.

The block is deliberately small, since the model has to write it for every
sample during reinforcement learning.

    <logic>
    p1: "the span from the question" |- Knight(mary)
    p2: "another span" |- Knight(mary) -> Knave(john)
    s1: p1,p2 |- Knave(john) [mp]
    s2: 12 * 3 = 36 [arith]
    concl: s2
    </logic>
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PREMISE_RE = re.compile(r'^\s*(p\d+)\s*:\s*"([^"]*)"\s*\|-\s*(.+?)\s*$')
STEP_RE = re.compile(r'^\s*(s\d+)\s*:\s*([ps\d,\s]*)\|-\s*(.+?)\s*\[(\w+)\]\s*$')
ARITH_RE = re.compile(r'^\s*(s\d+)\s*:\s*([-+*/(). \d]+)=\s*([-+.\d]+)\s*\[arith\]\s*$')
CONCL_RE = re.compile(r"^\s*concl\s*:\s*([ps]\d+)\s*$")
BLOCK_RE = re.compile(r"<logic>(.*?)</logic>", re.S)
RULES = {"mp", "mt", "and_i", "and_e", "or_e", "subst", "arith", "def"}
MAX_LINES = 40


@dataclass
class Result:
    has_block: bool = False
    grounded: bool = False
    steps_valid: bool = False
    load_bearing: bool = False
    non_trivial: bool = False
    n_steps: int = 0
    conclusion: str = ""
    reasons: list = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(
            (self.has_block, self.grounded, self.steps_valid,
             self.load_bearing, self.non_trivial)
        )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _arith_ok(expression: str, claimed: str) -> bool:
    if not re.fullmatch(r"[-+*/(). \d]+", expression):
        return False
    try:
        value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return abs(float(value) - float(claimed)) < 1e-6
    except Exception:
        return False


def check_grounded_logic(prompt: str, completion: str, stated_answer: str) -> Result:
    """Check one completion. `stated_answer` is what the model answered."""
    result = Result()
    match = BLOCK_RE.search(completion)
    if not match:
        result.reasons.append("no logic block")
        return result
    result.has_block = True
    lines = [ln for ln in match.group(1).splitlines() if ln.strip()]
    if not lines or len(lines) > MAX_LINES:
        result.reasons.append("block empty or too long")
        return result

    haystack = _norm(prompt)
    premises: dict[str, str] = {}
    steps: dict[str, str] = {}
    order: list[str] = []
    grounded = True
    steps_valid = True

    for line in lines:
        if CONCL_RE.match(line):
            continue
        premise = PREMISE_RE.match(line)
        if premise:
            label, quote, claim = premise.groups()
            premises[label] = claim
            order.append(label)
            if _norm(quote) not in haystack:
                grounded = False
                result.reasons.append(f"{label} quotes text absent from the prompt")
            continue
        arithmetic = ARITH_RE.match(line)
        if arithmetic:
            label, expression, value = arithmetic.groups()
            steps[label] = f"{expression.strip()}= {value}"
            order.append(label)
            result.n_steps += 1
            if not _arith_ok(expression, value):
                steps_valid = False
                result.reasons.append(f"{label} arithmetic does not check out")
            continue
        step = STEP_RE.match(line)
        if step:
            label, refs, claim, rule = step.groups()
            cited = [r.strip() for r in refs.split(",") if r.strip()]
            steps[label] = claim
            order.append(label)
            result.n_steps += 1
            if rule not in RULES:
                steps_valid = False
                result.reasons.append(f"{label} names an unknown rule {rule}")
            for ref in cited:
                if ref not in premises and ref not in steps:
                    steps_valid = False
                    result.reasons.append(f"{label} cites {ref}, which is not established")
                if ref == label:
                    steps_valid = False
                    result.reasons.append(f"{label} cites itself")
            if rule != "def" and not cited:
                steps_valid = False
                result.reasons.append(f"{label} cites nothing")
            continue
        steps_valid = False
        result.reasons.append(f"unparsable line: {line.strip()[:60]}")

    result.grounded = grounded
    result.steps_valid = steps_valid

    conclusion_line = [CONCL_RE.match(ln) for ln in lines]
    conclusion_line = [m.group(1) for m in conclusion_line if m]
    if not conclusion_line:
        result.reasons.append("no conclusion line")
        return result
    label = conclusion_line[-1]
    conclusion = steps.get(label) or premises.get(label, "")
    result.conclusion = conclusion
    if not conclusion:
        result.reasons.append("conclusion names a line that does not exist")
        return result

    # Load-bearing: the answer the model states has to appear in the concluding
    # line of the derivation, so a decorative block earns nothing.
    answer = _norm(stated_answer)
    result.load_bearing = bool(answer) and answer in _norm(conclusion)
    if not result.load_bearing:
        result.reasons.append("the concluding line does not carry the stated answer")

    # Non-trivial: the conclusion must come from a step, and that step must
    # rest on at least two distinct established lines, so assuming the answer
    # as a premise earns nothing.
    if label in premises:
        result.reasons.append("the conclusion is a premise")
    else:
        for line in lines:
            step = STEP_RE.match(line)
            if step and step.group(1) == label:
                cited = {r.strip() for r in step.group(2).split(",") if r.strip()}
                result.non_trivial = len(cited) >= 2
                if not result.non_trivial:
                    result.reasons.append("the concluding step rests on fewer than two lines")
            elif ARITH_RE.match(line) and ARITH_RE.match(line).group(1) == label:
                result.non_trivial = True
    return result
