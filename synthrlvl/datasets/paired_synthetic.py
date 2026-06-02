from __future__ import annotations

import hashlib
import os
import random
import re
import sys
from dataclasses import dataclass

from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from logic_engine import LogicEngine

from synthetic_dataset import LogicExample, ProofLine


PairedDatasetKind = Literal[
    "official_igsm",
    "igsm_arithmetic",
    "maze_navigation",
    "graph_traversal",
    "attribute_constraints",
    "mastermind_constraints",
    "constraint_satisfaction",
    "constraint_propagation",
]
PAIRED_DATASET_KINDS: tuple[str, ...] = (
    "official_igsm",
    "igsm_arithmetic",
    "maze_navigation",
    "graph_traversal",
    "attribute_constraints",
    "mastermind_constraints",
    "constraint_satisfaction",
    "constraint_propagation",
)


STATE_WORDS = (
    "amber",
    "cobalt",
    "ivory",
    "olive",
    "ruby",
    "slate",
    "coral",
    "lime",
    "pearl",
    "teal",
    "maple",
    "cedar",
    "hazel",
    "birch",
    "juniper",
    "willow",
    "laurel",
    "orchid",
    "violet",
    "poppy",
    "elm",
    "granite",
    "harbor",
    "meadow",
    "bramble",
    "fennel",
    "indigo",
    "ochre",
    "saffron",
    "umber",
    "silver",
    "onyx",
    "lilac",
    "marble",
    "spruce",
    "topaz",
    "acorn",
    "brook",
    "clover",
    "dune",
    "estate",
    "forest",
    "grove",
    "heather",
    "island",
    "jasper",
    "keystone",
    "linen",
    "moss",
    "nectar",
    "opal",
    "prairie",
    "quartz",
    "reed",
    "summit",
    "timber",
    "utopia",
    "valley",
    "walnut",
    "xenon",
    "yarrow",
    "zephyr",
    "aurora",
    "bison",
    "canopy",
    "drift",
    "eagle",
    "falcon",
    "garden",
    "haven",
    "iris",
    "jewel",
    "kelp",
    "lantern",
    "mint",
    "novel",
    "ridge",
    "cavern",
    "grotto",
    "basin",
    "spire",
    "terrace",
    "lagoon",
    "citadel",
    "gallery",
    "rotunda",
)

COLOR_WORDS = (
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "orange",
    "purple",
    "silver",
    "gold",
)


@dataclass(frozen=True)
class PairedGeneratorConfig:
    kind: PairedDatasetKind
    depth: int
    seed: int = 0
    branching_factor: int = 4
    distractor_ratio: float = 0.25
    max_operand: int = 20
    max_multiplier: int = 6
    operation_tokens: tuple[str, ...] = ("+", "-", "*")
    modulus: int | None = None
    official_igsm_repo_path: str | None = None
    official_igsm_max_edge: int | None = None
    official_igsm_perm_level: int = 5
    official_igsm_detail_level: int = 0
    official_igsm_p_format: str = "pq"
    candidate_count: int = 6

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.branching_factor < 1:
            raise ValueError("branching_factor must be >= 1")
        if self.kind not in set(PAIRED_DATASET_KINDS):
            raise ValueError(f"Unsupported paired dataset kind: {self.kind}")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: str | None = None
    line_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _EquationChain:
    original_var: str
    var: str
    expr: str
    result: int
    official_text: str
    semantic_name: str | None = None


def _igsm_semantic_constant_text(var: str, semantic_name: str) -> str:
    if semantic_name.startswith("intermediate value "):
        return f"{var} = {semantic_name}"
    return f"{var} = the number of each {semantic_name}"


def _igsm_semantic_proof_source(semantic_name: str) -> str:
    prefix = "intermediate value used to compute the number of each "
    if semantic_name.startswith(prefix):
        return f"the intermediate calculation for {semantic_name[len(prefix):]}"
    return f"the definition of {semantic_name}"


class PairedSyntheticGenerator:
    """Generate paired natural-language and formal-logic reasoning traces.

    The public output is the repo's existing ``LogicExample`` schema. New task
    families share the SFT formatter, materialized loader, and logic verifier.
    """

    def __init__(self, config: PairedGeneratorConfig):
        self.config = config

    def _rng(self, index: int) -> random.Random:
        h = hashlib.sha256(f"{self.config.seed}|{self.config.kind}|{index}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def generate(self, index: int) -> LogicExample:
        if self.config.kind == "official_igsm":
            return self._generate_official_igsm(index)
        if self.config.kind == "igsm_arithmetic":
            return self._generate_register_arithmetic(index)
        if self.config.kind in {"maze_navigation", "graph_traversal"}:
            return self._generate_maze_navigation(index)
        if self.config.kind in {
            "attribute_constraints",
            "mastermind_constraints",
            "constraint_satisfaction",
            "constraint_propagation",
        }:
            return self._generate_attribute_constraints(index)
        raise ValueError(f"Unsupported paired dataset kind: {self.config.kind}")

    # ------------------------------------------------------------------
    # Official iGSM-backed arithmetic.
    # ------------------------------------------------------------------

    def _generate_official_igsm(self, index: int) -> LogicExample:
        id_gen = self._sample_official_igsm(index)
        problem = id_gen.problem
        chains = self._equation_chains_from_igsm_solution(problem.solution)
        if not chains:
            raise RuntimeError("official iGSM solution produced no parseable equation chains")

        premise_formulas = [f"{chain.var} = {chain.expr}" for chain in chains]
        premises_fol = [f"{idx}. {formula}" for idx, formula in enumerate(premise_formulas, start=1)]
        premises_nl = [
            f"{idx}. {sentence.strip()}." if not str(sentence).strip().endswith("?") else f"{idx}. {sentence.strip()}"
            for idx, sentence in enumerate(getattr(problem, "problem", [])[:-1], start=1)
        ]
        if not premises_nl:
            text = self._decode_igsm_tokens(id_gen.prob_token)
            premises_nl = [f"1. {text.strip()}"]

        proof_fol, proof_nl, final_var_line, known_values = self._prove_igsm_chains(chains)
        answer = str(int(getattr(problem, "ans", known_values.get(chains[-1].original_var, 0))))
        conclusion = proof_fol[-1].split(". ", 1)[1].split(" ; ", 1)[0]
        question = str(getattr(problem, "problem", ["What is the answer?"])[-1]).strip()

        constants = [
            _igsm_semantic_constant_text(chain.var, chain.semantic_name)
            if chain.semantic_name
            else f"{chain.var} = quantity named {chain.original_var}"
            for chain in sorted(chains, key=lambda item: item.original_var)
        ]

        return LogicExample(
            constants=constants,
            predicates=[],
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol=f"What is the value of the queried official iGSM quantity? Prove {conclusion}.",
            question_nl=question,
            answer=answer,
            metadata={
                "dataset_family": "official_igsm",
                "depth": int(self.config.depth),
                "official_n_op": int(getattr(problem, "n_op", self.config.depth)),
                "official_problem_text": self._decode_igsm_tokens(id_gen.prob_token),
                "official_solution_text": self._decode_igsm_tokens(id_gen.sol_token),
                "official_answer_text": self._decode_igsm_tokens(id_gen.ans_token).strip(),
                "equation_chains": [chain.__dict__ for chain in chains],
                "gold_answer": answer,
                "modulus": 23,
                "logic_trace_valid": True,
            },
        )

    def _sample_official_igsm(self, index: int):
        repo = self._official_igsm_repo()
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from data_gen.pretrain.id_gen import IdGen  # type: ignore
        from tools.tools import fix_seed  # type: ignore

        depth = int(self.config.depth)
        max_op = max(depth, 5)
        max_edge = int(self.config.official_igsm_max_edge or max(depth + 5, 8))
        last_error: Exception | None = None
        for attempt in range(100):
            try:
                fix_seed(int(self.config.seed) + int(index) * 9973 + attempt)
                id_gen = IdGen(
                    max_op=max_op,
                    max_edge=max_edge,
                    op=depth,
                    perm_level=int(self.config.official_igsm_perm_level),
                    detail_level=int(self.config.official_igsm_detail_level),
                )
                id_gen.gen_prob([i for i in range(23)], p_format=str(self.config.official_igsm_p_format))
                if int(getattr(id_gen.problem, "n_op", depth)) == depth:
                    return id_gen
            except Exception as exc:  # pragma: no cover - depends on official generator internals.
                last_error = exc
        raise RuntimeError(f"Could not sample official iGSM example with op={depth}: {last_error}")

    def _official_igsm_repo(self) -> Path:
        if self.config.official_igsm_repo_path:
            repo = Path(self.config.official_igsm_repo_path)
        elif os.environ.get("IGSM_REPO_PATH"):
            repo = Path(os.environ["IGSM_REPO_PATH"])
        elif os.environ.get("WORK"):
            repo = Path(os.environ["WORK"]) / "codex_research/iGSM"
        else:
            repo = Path("iGSM")
        if not (repo / "data_gen/pretrain/id_gen.py").exists():
            raise FileNotFoundError(
                f"Could not find facebookresearch/iGSM at {repo}. Set IGSM_REPO_PATH or official_igsm_repo_path."
            )
        return repo

    def _decode_igsm_tokens(self, token_ids: Sequence[int]) -> str:
        repo = self._official_igsm_repo()
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from tools.tools import tokenizer  # type: ignore

        return tokenizer.decode(list(token_ids))

    @staticmethod
    def _equation_chains_from_igsm_solution(solution_lines: Sequence[str]) -> list[_EquationChain]:
        raw_chains: list[dict[str, Any]] = []
        semantic_by_var: dict[str, str] = {}
        for raw_line in solution_lines:
            text = str(raw_line).strip().rstrip(".")
            if not text:
                continue
            for match in re.finditer(r"\bDefine\s+(.+?)\s+as\s+([A-Za-z])\b", text):
                semantic_by_var[match.group(2)] = match.group(1).strip()
            clauses = [part.strip() for part in text.split(";") if part.strip()]
            current_semantic: str | None = None
            for clause in clauses:
                define_match = re.search(r"\bDefine\s+(.+?)\s+as\s+([A-Za-z])\b", clause)
                if define_match:
                    current_semantic = define_match.group(1).strip()
                if clause.lower().startswith("define ") and ";" not in clause:
                    # Keep only the assignment part if the define clause also contains `so`.
                    if " so " not in clause:
                        continue
                if clause.startswith("so "):
                    clause = clause[3:].strip()
                if " so " in clause:
                    clause = clause.split(" so ", 1)[1].strip()
                if "=" not in clause:
                    continue
                parts = [part.strip() for part in clause.split("=") if part.strip()]
                if len(parts) < 2 or not _looks_like_igsm_var(parts[0]):
                    continue
                try:
                    result = int(parts[-1]) % 23
                except ValueError:
                    continue
                original_var = parts[0]
                semantic_name = semantic_by_var.get(original_var)
                if semantic_name is None and current_semantic:
                    semantic_name = f"intermediate value used to compute the number of each {current_semantic}"
                raw_chains.append(
                    {
                        "original_var": original_var,
                        "raw_expr": parts[1],
                        "result": result,
                        "official_text": clause,
                        "semantic_name": semantic_name,
                    }
                )
        var_map = _igsm_safe_symbol_map([str(chain["original_var"]) for chain in raw_chains])
        return [
            _EquationChain(
                original_var=str(chain["original_var"]),
                var=var_map[str(chain["original_var"])],
                expr=_normalize_igsm_expr(str(chain["raw_expr"]), var_map=var_map),
                result=int(chain["result"]),
                official_text=str(chain["official_text"]),
                semantic_name=chain["semantic_name"],
            )
            for chain in raw_chains
        ]

    def _prove_igsm_chains(self, chains: Sequence[_EquationChain]) -> tuple[list[str], list[str], int | None, dict[str, int]]:
        proof_fol: list[str] = []
        proof_nl: list[str] = []
        known_lines: dict[str, int] = {}
        known_values: dict[str, int] = {}
        symbol_by_original = {chain.original_var: chain.var for chain in chains}
        original_by_symbol = {symbol: original for original, symbol in symbol_by_original.items()}
        next_line = len(chains) + 1
        final_line: int | None = None

        for premise_idx, chain in enumerate(chains, start=1):
            current_expr = chain.expr
            current_formula = f"{chain.var} = {current_expr}"
            proof_fol.append(ProofLine(next_line, current_formula, f"R,{premise_idx}").render())
            if chain.semantic_name:
                proof_nl.append(
                    f"{next_line}. From {_igsm_semantic_proof_source(chain.semantic_name)} ({chain.var}), "
                    f"{chain.var} equals {current_expr}."
                )
            else:
                proof_nl.append(f"{next_line}. From the official iGSM relation, {chain.var} equals {current_expr}.")
            current_line = next_line
            next_line += 1

            for symbol in _expr_vars(current_expr):
                var = original_by_symbol.get(symbol, symbol)
                if var not in known_lines:
                    continue
                value = str(known_values[var])
                current_expr = _replace_token(current_expr, symbol, value)
                current_formula = f"{chain.var} = {current_expr}"
                proof_fol.append(ProofLine(next_line, current_formula, f"=E,{known_lines[var]},{current_line}").render())
                proof_nl.append(f"{next_line}. Substitute {symbol} = {value} into the current expression.")
                current_line = next_line
                next_line += 1

            if _safe_eval_mod23(current_expr) is not None and current_expr.strip() != str(chain.result):
                current_formula = f"{chain.var} = {chain.result}"
                proof_fol.append(ProofLine(next_line, current_formula, f"MOD23,{current_line}").render())
                proof_nl.append(f"{next_line}. Evaluate the arithmetic modulo 23 to get {chain.var} = {chain.result}.")
                current_line = next_line
                next_line += 1

            if current_formula != f"{chain.var} = {chain.result}":
                # Covers direct variable aliases where substitution already produced the final numeral.
                current_formula = f"{chain.var} = {chain.result}"
                proof_fol.append(ProofLine(next_line, current_formula, f"MOD23,{current_line}").render())
                proof_nl.append(f"{next_line}. Reduce the value modulo 23 to get {chain.var} = {chain.result}.")
                current_line = next_line
                next_line += 1

            known_lines[chain.original_var] = current_line
            known_values[chain.original_var] = chain.result
            final_line = current_line

        return proof_fol, proof_nl, final_line, known_values

    # ------------------------------------------------------------------
    # Compact register arithmetic retained for backward compatibility.
    # ------------------------------------------------------------------

    def _generate_register_arithmetic(self, index: int) -> LogicExample:
        rng = self._rng(index)
        depth = int(self.config.depth)
        operations: list[tuple[str, int, int, int]] = []
        current = rng.randint(0, min(20, self.config.max_operand))
        if self.config.modulus is not None:
            current %= int(self.config.modulus)
        start_value = current
        for _ in range(depth):
            op, operand, result = self._sample_operation(rng, current)
            operations.append((op, operand, current, result))
            current = result

        premises_fol: list[str] = [f"1. x0 = {start_value}"]
        premises_nl: list[str] = [f"1. The initial value x0 is {start_value}."]
        for step, (op, operand, _prev, _result) in enumerate(operations, start=1):
            var = f"x{step}"
            prev_var = f"x{step - 1}"
            premises_fol.append(f"{len(premises_fol) + 1}. {var} = {self._arith_rhs(prev_var, op, operand)}")
            premises_nl.append(
                f"{len(premises_nl) + 1}. {var} is obtained by applying {prev_var} {self._op_word(op)} {operand}."
            )

        proof_fol: list[str] = []
        proof_nl: list[str] = []
        next_line = len(premises_fol) + 1
        proof_fol.append(ProofLine(next_line, f"x0 = {start_value}", "R,1").render())
        proof_nl.append(f"{next_line}. x0 is {start_value}.")
        prev_value_line = next_line
        next_line += 1
        for step, (op, operand, prev_value, result) in enumerate(operations, start=1):
            var = f"x{step}"
            premise_line = step + 1
            substituted = f"{var} = {self._arith_rhs(str(prev_value), op, operand)}"
            proof_fol.append(ProofLine(next_line, substituted, f"=E,{prev_value_line},{premise_line}").render())
            proof_nl.append(
                f"{next_line}. Since x{step - 1} is {prev_value}, {var} is {prev_value} {self._op_word(op)} {operand}."
            )
            substituted_line = next_line
            next_line += 1
            proof_fol.append(ProofLine(next_line, f"{var} = {result}", f"ARITH,{substituted_line}").render())
            proof_nl.append(f"{next_line}. Therefore {var} is {result}.")
            prev_value_line = next_line
            next_line += 1

        answer = str(current)
        return LogicExample(
            constants=[f"x{idx} = register {idx}" for idx in range(depth + 1)],
            predicates=[],
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol=f"What is the value of x{depth}?",
            question_nl=f"What is the final value after {depth} arithmetic operations?",
            answer=answer,
            metadata={
                "dataset_family": "igsm_arithmetic",
                "depth": depth,
                "start_value": start_value,
                "operations": [
                    {"op": op, "operand": operand, "input": prev, "output": result}
                    for op, operand, prev, result in operations
                ],
                "gold_answer": answer,
                "logic_trace_valid": True,
            },
        )

    def _sample_operation(self, rng: random.Random, current: int) -> tuple[str, int, int]:
        op = rng.choice(tuple(self.config.operation_tokens))
        if op not in {"+", "-", "*"}:
            raise ValueError(f"Unsupported operation token: {op}")
        if op == "*":
            operand = rng.randint(2, int(self.config.max_multiplier))
        else:
            operand = rng.randint(1, int(self.config.max_operand))
        result = {"+": current + operand, "-": current - operand, "*": current * operand}[op]
        if self.config.modulus is not None:
            result %= int(self.config.modulus)
        return op, operand, result

    @staticmethod
    def _op_word(op: str) -> str:
        return {"+": "plus", "-": "minus", "*": "times"}[op]

    @staticmethod
    def _arith_rhs(left: str, op: str, operand: int) -> str:
        if op == "-":
            return f"{left} + (-{operand})"
        return f"{left} {op} {operand}"

    # ------------------------------------------------------------------
    # Maze navigation: key-constrained graph traversal.
    # ------------------------------------------------------------------

    def _generate_maze_navigation(self, index: int) -> LogicExample:
        rng = self._rng(index)
        depth = int(self.config.depth)
        width = max(2, min(int(self.config.branching_factor), 4))
        needed = depth + 1 + depth * width + width + 8
        names = rng.sample(_maze_room_bank(needed), needed)
        room_path = names[: depth + 1]
        decoy_rooms = names[depth + 1 : depth + 1 + depth * width]
        treasure_decoys = names[depth + 1 + depth * width : depth + 1 + depth * width + width]
        start = room_path[0]
        target = room_path[-1]
        key_bank = _maze_key_bank(max(depth + width + 2, 6))
        key_path = rng.sample(key_bank, depth + 1)

        treasure_rooms = [target, *treasure_decoys]
        rng.shuffle(treasure_rooms)

        premises_fol: list[str] = [f"1. At0({start})", f"2. Have0({key_path[0]})"]
        premises_nl: list[str] = [
            f"1. The explorer starts in room {start}.",
            f"2. The explorer initially holds the {key_path[0]} key.",
        ]
        transition_refs: dict[int, tuple[int, int, int, int]] = {}
        blocked_edges: list[dict[str, str | int]] = []
        for step in range(depth):
            src = room_path[step]
            dst = room_path[step + 1]
            key = key_path[step]
            next_key = key_path[step + 1]

            edge_specs: list[tuple[str, str, bool]] = [(key, dst, True)]
            wrong_keys = [candidate for candidate in key_bank if candidate != key]
            rng.shuffle(wrong_keys)
            for offset, wrong_key in enumerate(wrong_keys[:width]):
                decoy = decoy_rooms[step * width + offset]
                edge_specs.append((wrong_key, decoy, False))
                blocked_edges.append(
                    {
                        "step": step,
                        "from_room": src,
                        "required_key": wrong_key,
                        "to_room": decoy,
                    }
                )
            rng.shuffle(edge_specs)

            for required_key, dst_room, is_gold in edge_specs:
                door_line = len(premises_fol) + 1
                premises_fol.append(f"{door_line}. Door({src},{required_key},{dst_room})")
                premises_nl.append(
                    f"{door_line}. There is a door from {src} to {dst_room} that requires the {required_key} key."
                )
                rule_line = len(premises_fol) + 1
                antecedent = _nested_and(
                    [
                        f"At{step}({src})",
                        f"Have{step}({required_key})",
                        f"Door({src},{required_key},{dst_room})",
                    ]
                )
                premises_fol.append(f"{rule_line}. {antecedent} -> At{step + 1}({dst_room})")
                premises_nl.append(
                    f"{rule_line}. If the explorer is in {src} after {step} moves, has the {required_key} key, "
                    f"and the matching door leads to {dst_room}, then {dst_room} is reachable after {step + 1} moves."
                )
                if is_gold:
                    find_line = len(premises_fol) + 1
                    premises_fol.append(f"{find_line}. Finds({dst},{next_key})")
                    premises_nl.append(f"{find_line}. Room {dst} contains the {next_key} key.")
                    key_rule_line = len(premises_fol) + 1
                    premises_fol.append(
                        f"{key_rule_line}. At{step + 1}({dst}) & Finds({dst},{next_key}) -> Have{step + 1}({next_key})"
                    )
                    premises_nl.append(
                        f"{key_rule_line}. If the explorer reaches {dst} after {step + 1} moves and {dst} contains "
                        f"the {next_key} key, then the explorer has the {next_key} key after {step + 1} moves."
                    )
                    transition_refs[step] = (door_line, rule_line, find_line, key_rule_line)

        treasure_lines: dict[str, int] = {}
        for room in treasure_rooms:
            line = len(premises_fol) + 1
            premises_fol.append(f"{line}. Treasure({room})")
            premises_nl.append(f"{line}. Room {room} contains a marked treasure.")
            treasure_lines[room] = line
        found_rule_lines: dict[str, int] = {}
        for room in treasure_rooms:
            found_rule_line = len(premises_fol) + 1
            premises_fol.append(f"{found_rule_line}. At{depth}({room}) & Treasure({room}) -> Found({room})")
            premises_nl.append(
                f"{found_rule_line}. If room {room} is reachable after exactly {depth} key-constrained moves and contains "
                f"a treasure, then the treasure in {room} is found."
            )
            found_rule_lines[room] = found_rule_line

        proof_fol: list[str] = []
        proof_nl: list[str] = []
        next_line = len(premises_fol) + 1
        proof_fol.append(ProofLine(next_line, f"At0({start})", "R,1").render())
        proof_nl.append(f"{next_line}. The explorer is at {start} after 0 moves.")
        at_line = next_line
        next_line += 1
        proof_fol.append(ProofLine(next_line, f"Have0({key_path[0]})", "R,2").render())
        proof_nl.append(f"{next_line}. The explorer has the {key_path[0]} key after 0 moves.")
        key_line = next_line
        next_line += 1

        for step in range(depth):
            src = room_path[step]
            dst = room_path[step + 1]
            key = key_path[step]
            next_key = key_path[step + 1]
            door_line, rule_line, find_line, key_rule_line = transition_refs[step]

            door_formula = f"Door({src},{key},{dst})"
            proof_fol.append(ProofLine(next_line, door_formula, f"R,{door_line}").render())
            proof_nl.append(f"{next_line}. The door from {src} to {dst} requires the {key} key.")
            door_proof_line = next_line
            next_line += 1

            at_key = f"At{step}({src}) & Have{step}({key})"
            proof_fol.append(ProofLine(next_line, at_key, f"∧I,{at_line},{key_line}").render())
            proof_nl.append(f"{next_line}. The explorer is at {src} and has the required {key} key.")
            at_key_line = next_line
            next_line += 1

            transition_condition = f"{at_key} & {door_formula}"
            proof_fol.append(ProofLine(next_line, transition_condition, f"∧I,{at_key_line},{door_proof_line}").render())
            proof_nl.append(f"{next_line}. Combine location, key, and door constraint.")
            transition_condition_line = next_line
            next_line += 1

            at_formula = f"At{step + 1}({dst})"
            proof_fol.append(ProofLine(next_line, at_formula, f"->E,{rule_line},{transition_condition_line}").render())
            proof_nl.append(f"{next_line}. Therefore {dst} is reachable after {step + 1} moves.")
            at_line = next_line
            next_line += 1

            find_formula = f"Finds({dst},{next_key})"
            proof_fol.append(ProofLine(next_line, find_formula, f"R,{find_line}").render())
            proof_nl.append(f"{next_line}. Room {dst} contains the {next_key} key.")
            find_proof_line = next_line
            next_line += 1

            key_condition = f"{at_formula} & {find_formula}"
            proof_fol.append(ProofLine(next_line, key_condition, f"∧I,{at_line},{find_proof_line}").render())
            proof_nl.append(f"{next_line}. Combine reaching {dst} with finding the key there.")
            key_condition_line = next_line
            next_line += 1

            have_formula = f"Have{step + 1}({next_key})"
            proof_fol.append(ProofLine(next_line, have_formula, f"->E,{key_rule_line},{key_condition_line}").render())
            proof_nl.append(f"{next_line}. Therefore the explorer has the {next_key} key after {step + 1} moves.")
            key_line = next_line
            next_line += 1

        proof_fol.append(ProofLine(next_line, f"Treasure({target})", f"R,{treasure_lines[target]}").render())
        proof_nl.append(f"{next_line}. Room {target} contains a marked treasure.")
        treasure_proof_line = next_line
        next_line += 1
        conj = f"At{depth}({target}) & Treasure({target})"
        proof_fol.append(ProofLine(next_line, conj, f"∧I,{at_line},{treasure_proof_line}").render())
        proof_nl.append(f"{next_line}. The reachable room {target} contains a treasure.")
        conj_line = next_line
        next_line += 1
        proof_fol.append(ProofLine(next_line, f"Found({target})", f"->E,{found_rule_lines[target]},{conj_line}").render())
        proof_nl.append(f"{next_line}. Therefore the found treasure is in {target}.")

        return LogicExample(
            constants=[f"{room} = maze room {room}" for room in sorted(set(room_path) | set(decoy_rooms) | set(treasure_decoys))]
            + [f"{key} = maze key {key}" for key in sorted(set(key_bank))],
            predicates=[
                "AtN(x): the explorer can be at room x after N moves",
                "HaveN(x): the explorer has key x after N moves",
                "Door(x,y,z): there is a door from room x to room z requiring key y",
                "Finds(x,y): room x contains key y",
                "Treasure(x): room x contains a marked treasure",
                "Found(x): the reachable marked treasure is in room x",
            ],
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol=f"Which marked treasure room is reachable after exactly {depth} key-constrained moves?",
            question_nl=(
                f"The rooms form a locked maze. The explorer may use only doors whose key they currently hold, "
                f"and entering a room may reveal the next key. Which marked treasure room is reachable after exactly {depth} moves?"
            ),
            answer=target,
            metadata={
                "dataset_family": "maze_navigation",
                "task_structure": "keyed_constrained_graph",
                "depth": depth,
                "start": start,
                "gold_path": room_path,
                "key_path": key_path,
                "blocked_edges": blocked_edges,
                "treasure_rooms": treasure_rooms,
                "unreachable_treasure_rooms": treasure_decoys,
                "requires_key_tracking": True,
                "solution_rule_for_all_treasures": True,
                "gold_answer": target,
                "logic_trace_valid": True,
            },
        )

    # ------------------------------------------------------------------
    # Attribute constraints: multi-input slot-value constraint propagation.
    # ------------------------------------------------------------------

    def _generate_attribute_constraints(self, index: int) -> LogicExample:
        rng = self._rng(index)
        length = max(2, int(self.config.depth) // 2 + 2)
        palette = _attribute_value_bank(max(length + 6, int(self.config.branching_factor) * 4 + 8))
        values = tuple(rng.sample(palette, length))
        slot_ids = [f"s{idx}" for idx in range(length)]

        premises_fol: list[str] = []
        premises_nl: list[str] = []
        value_lines: dict[int, int] = {}
        constraint_lines: dict[int, int] = {}
        rule_lines: dict[int, int] = {}

        base_count = min(2, length)
        for pos in range(base_count):
            line = len(premises_fol) + 1
            premises_fol.append(f"{line}. Value({slot_ids[pos]},{values[pos]})")
            premises_nl.append(f"{line}. {slot_ids[pos]} has value {values[pos]}.")
            value_lines[pos] = line

        decoy_count = max(1, min(int(self.config.branching_factor), 2))
        derivation_rules: list[dict[str, str | int]] = []
        decoy_rules: list[dict[str, str | int]] = []
        for pos in range(base_count, length):
            dep_window_start = max(0, pos - max(3, int(self.config.branching_factor)))
            dep_a = rng.randrange(dep_window_start, pos - 1)
            dep_b = pos - 1
            slot_a = slot_ids[dep_a]
            slot_b = slot_ids[dep_b]
            slot = slot_ids[pos]
            color_a = values[dep_a]
            color_b = values[dep_b]
            color = values[pos]

            line = len(premises_fol) + 1
            premises_fol.append(f"{line}. Constraint({slot_a},{color_a},{slot_b},{color_b},{slot},{color})")
            premises_nl.append(
                f"{line}. The joint constraint says: if {slot_a} is {color_a} and {slot_b} is {color_b}, then {slot} is {color}."
            )
            constraint_lines[pos] = line

            line = len(premises_fol) + 1
            constraint_formula = f"Constraint({slot_a},{color_a},{slot_b},{color_b},{slot},{color})"
            antecedent = _nested_and(
                [
                    f"Value({slot_a},{color_a})",
                    f"Value({slot_b},{color_b})",
                    constraint_formula,
                ]
            )
            premises_fol.append(f"{line}. {antecedent} -> Value({slot},{color})")
            premises_nl.append(
                f"{line}. If both prerequisite slot values hold and the matching joint constraint is present, then {slot} has {color}."
            )
            rule_lines[pos] = line
            derivation_rules.append(
                {
                    "target_index": pos,
                    "dep_a": dep_a,
                    "dep_b": dep_b,
                    "slot_a": slot_a,
                    "value_a": color_a,
                    "slot_b": slot_b,
                    "value_b": color_b,
                    "target_slot": slot,
                    "target_value": color,
                }
            )

            wrong_pairs: list[tuple[str, str]] = []
            wrong_a_values = [candidate for candidate in palette if candidate != color_a]
            wrong_b_values = [candidate for candidate in palette if candidate != color_b]
            for wrong_b in wrong_b_values:
                wrong_pairs.append((color_a, wrong_b))
            for wrong_a in wrong_a_values:
                wrong_pairs.append((wrong_a, color_b))
            for wrong_a in wrong_a_values:
                for wrong_b in wrong_b_values:
                    wrong_pairs.append((wrong_a, wrong_b))
            rng.shuffle(wrong_pairs)
            for wrong_a, wrong_b in wrong_pairs[:decoy_count]:
                wrong_target_choices = [candidate for candidate in palette if candidate != color]
                wrong_target = rng.choice(wrong_target_choices)
                line = len(premises_fol) + 1
                premises_fol.append(f"{line}. Constraint({slot_a},{wrong_a},{slot_b},{wrong_b},{slot},{wrong_target})")
                premises_nl.append(
                    f"{line}. A decoy joint constraint says: if {slot_a} is {wrong_a} and {slot_b} is {wrong_b}, "
                    f"then {slot} is {wrong_target}."
                )
                line = len(premises_fol) + 1
                decoy_constraint = f"Constraint({slot_a},{wrong_a},{slot_b},{wrong_b},{slot},{wrong_target})"
                decoy_antecedent = _nested_and(
                    [
                        f"Value({slot_a},{wrong_a})",
                        f"Value({slot_b},{wrong_b})",
                        decoy_constraint,
                    ]
                )
                premises_fol.append(f"{line}. {decoy_antecedent} -> Value({slot},{wrong_target})")
                premises_nl.append(
                    f"{line}. If the decoy prerequisite values held and the decoy constraint applied, then {slot} would be {wrong_target}."
                )
                decoy_rules.append(
                    {
                        "target_index": pos,
                        "slot_a": slot_a,
                        "value_a": wrong_a,
                        "slot_b": slot_b,
                        "value_b": wrong_b,
                        "target_slot": slot,
                        "target_value": wrong_target,
                    }
                )

        proof_fol: list[str] = []
        proof_nl: list[str] = []
        next_line = len(premises_fol) + 1
        value_proof_lines: dict[int, int] = {}

        for pos in range(base_count):
            slot = slot_ids[pos]
            color = values[pos]
            proof_fol.append(ProofLine(next_line, f"Value({slot},{color})", f"R,{value_lines[pos]}").render())
            proof_nl.append(f"{next_line}. Therefore {slot} has value {color}.")
            value_proof_lines[pos] = next_line
            next_line += 1

        for pos in range(base_count, length):
            rule = derivation_rules[pos - base_count]
            dep_a = int(rule["dep_a"])
            dep_b = pos - 1
            slot_a = slot_ids[dep_a]
            slot_b = slot_ids[dep_b]
            slot = slot_ids[pos]
            color_a = values[dep_a]
            color_b = values[dep_b]
            color = values[pos]

            constraint_formula = f"Constraint({slot_a},{color_a},{slot_b},{color_b},{slot},{color})"
            proof_fol.append(ProofLine(next_line, constraint_formula, f"R,{constraint_lines[pos]}").render())
            proof_nl.append(
                f"{next_line}. The applicable joint constraint maps {slot_a}={color_a} and {slot_b}={color_b} to {slot}={color}."
            )
            constraint_proof_line = next_line
            next_line += 1

            prereq_pair = f"Value({slot_a},{color_a}) & Value({slot_b},{color_b})"
            proof_fol.append(ProofLine(next_line, prereq_pair, f"∧I,{value_proof_lines[dep_a]},{value_proof_lines[dep_b]}").render())
            proof_nl.append(f"{next_line}. Combine the two prerequisite slot values.")
            prereq_pair_line = next_line
            next_line += 1

            antecedent = f"{prereq_pair} & {constraint_formula}"
            proof_fol.append(ProofLine(next_line, antecedent, f"∧I,{prereq_pair_line},{constraint_proof_line}").render())
            proof_nl.append(f"{next_line}. Combine the prerequisites with the applicable joint constraint.")
            antecedent_line = next_line
            next_line += 1

            value_formula = f"Value({slot},{color})"
            proof_fol.append(ProofLine(next_line, value_formula, f"->E,{rule_lines[pos]},{antecedent_line}").render())
            proof_nl.append(f"{next_line}. Therefore {slot} has value {color}.")
            value_proof_lines[pos] = next_line
            next_line += 1

        value_formulas = [f"Value({slot_ids[pos]},{values[pos]})" for pos in range(length)]
        conclusion = value_formulas[0]
        conclusion_line = value_proof_lines[0]
        for pos in range(1, length):
            conclusion = f"{conclusion} & {value_formulas[pos]}"
            proof_fol.append(ProofLine(next_line, conclusion, f"∧I,{conclusion_line},{value_proof_lines[pos]}").render())
            proof_nl.append(f"{next_line}. Combine the solved values through {slot_ids[pos]}.")
            conclusion_line = next_line
            next_line += 1

        answer = _code_text(values)
        return LogicExample(
            constants=[f"{slot} = attribute slot {idx}" for idx, slot in enumerate(slot_ids)]
            + [f"{color} = attribute value {color}" for color in sorted(set(palette))],
            predicates=[
                "Value(x,y): slot x has value y",
                "Constraint(x,y,z,w,u,v): values y and w at slots x and z jointly force value v at slot u",
            ],
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol="Which values fill all constrained slots?",
            question_nl="Starting from the given slot values, apply the joint constraints. Which values fill all slots?",
            answer=answer,
            metadata={
                "dataset_family": "attribute_constraints",
                "task_structure": "multi_input_slot_constraint_dag",
                "depth": int(self.config.depth),
                "slot_count": length,
                "base_slot_count": base_count,
                "palette": palette,
                "slots": [
                    {"slot": slot_ids[pos], "value": values[pos]}
                    for pos in range(length)
                ],
                "constraints": derivation_rules,
                "decoy_constraints": decoy_rules,
                "gold_answer": answer,
                "logic_trace_valid": True,
                "grounded_validity_supported": True,
            },
        )



def _looks_like_igsm_var(text: str) -> bool:
    return re.fullmatch(r"[A-Za-z]", text.strip()) is not None


def _official_var_name(var: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", var.strip())
    return cleaned


def _is_safe_bare_igsm_symbol(symbol: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]", symbol))


def _igsm_safe_symbol_map(original_vars: Sequence[str]) -> dict[str, str]:
    used: set[str] = set()
    mapping: dict[str, str] = {}
    pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    for original in original_vars:
        if original in mapping:
            continue
        candidates = [original] + pool
        for candidate in candidates:
            if _is_safe_bare_igsm_symbol(candidate) and candidate not in used:
                mapping[original] = candidate
                used.add(candidate)
                break
        else:
            raise RuntimeError("official iGSM example uses more one-letter variables than the safe symbol pool")
    return mapping


def _maze_key_bank(size: int) -> list[str]:
    keys = list(COLOR_WORDS)
    while len(keys) < int(size):
        keys.append(f"key_{len(keys):02d}")
    return keys[: int(size)]


def _maze_room_bank(size: int) -> list[str]:
    rooms = list(STATE_WORDS)
    while len(rooms) < int(size):
        rooms.append(f"room_{len(rooms):03d}")
    return rooms[: int(size)]


def _attribute_value_bank(size: int) -> list[str]:
    values = list(COLOR_WORDS)
    while len(values) < int(size):
        values.append(f"v{len(values)}")
    return values[: int(size)]


def _normalize_igsm_expr(expr: str, *, var_map: dict[str, str] | None = None) -> str:
    expr = expr.strip()
    if var_map is None:
        var_map = {var: _official_var_name(var) for var in re.findall(r"\b([A-Za-z])\b", expr)}
    expr = re.sub(r"\b([A-Za-z])\b", lambda m: var_map.get(m.group(1), _official_var_name(m.group(1))), expr)
    return expr


def _expr_vars(expr: str) -> list[str]:
    vars_seen: list[str] = []
    for match in re.finditer(r"\b([A-Za-z])\b", expr):
        var = match.group(1)
        if var not in vars_seen:
            vars_seen.append(var)
    return vars_seen


def _replace_token(expr: str, token: str, value: str) -> str:
    return re.sub(rf"\b{re.escape(token)}\b", value, expr)


def _safe_eval_mod23(expr: str) -> int | None:
    if not re.fullmatch(r"[0-9+\-*/() ]+", expr.strip()):
        return None
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None
    if not isinstance(value, int):
        return None
    return int(value) % 23


def _nested_and(formulas: Sequence[str]) -> str:
    if not formulas:
        raise ValueError("cannot build conjunction from empty formula list")
    current = formulas[0]
    for formula in formulas[1:]:
        current = f"{current} & {formula}"
    return current




def _code_text(code: Sequence[str]) -> str:
    return "-".join(code)




def validate_logic_example(example: LogicExample, *, citation_free: bool = False) -> ValidationResult:
    engine = LogicEngine()
    premises = "\n".join(example.premises_fol)
    proof = "\n".join(example.proof_fol)
    conclusion = example.proof_fol[-1].split(". ", 1)[1].split(" ; ", 1)[0].strip()
    if citation_free:
        report = engine.analyze_proof_citation_free(premises=premises, conclusion=conclusion, proof=proof)
    else:
        report = engine.analyze_proof(premises=premises, conclusion=conclusion, proof=proof)
    return ValidationResult(
        ok=bool(report.ok),
        error=report.error,
        line_errors=tuple(
            f"{line.line_number}: {line.error or line.syntax_error}"
            for line in report.lines
            if not line.valid
        ),
    )


def finite_paired_examples(config: PairedGeneratorConfig, n: int, start_index: int = 0) -> Iterable[dict[str, Any]]:
    generator = PairedSyntheticGenerator(config)
    for index in range(start_index, start_index + n):
        yield generator.generate(index).to_dict()
