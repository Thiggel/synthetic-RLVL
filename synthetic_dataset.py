
"""
Synthetic typed logic dataset generator.

Core properties
---------------
- unary predicates only
- implication and conjunction in antecedent
- typed, natural-sounding attribute families
- lowercase constants a-z
- uppercase predicates A-Z
- exact proof syntax with numbered premises / proof lines and rule references
- structured rule templates with paired FOL and natural-language renderers
- Hugging Face friendly on-the-fly generation
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Any
import hashlib
import random
import json


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class DatasetConfig:
    depth: int
    distractor_ratio: float = 0.5
    difficulty: str = "standard"
    branching_factor: int | None = None
    decoy_chains: int | None = None
    near_miss_ratio: float | None = None
    side_chain_depth: int | None = None
    entity_decoy_ratio: float | None = None
    answer_decoy_ratio: float | None = None
    shortcut_rate: float = 0.0
    require_unique_solution: bool = True
    min_entities: int = 4
    max_entities: int = 8
    num_attribute_values: int = 8
    seed: int = 0

    def __post_init__(self):
        if self.difficulty not in {"standard", "hard_v1", "hard_v2", "hard_v3", "hard_v5", "hard_fsa", "hard_fsa_schema"}:
            raise ValueError("difficulty must be one of: standard, hard_v1, hard_v2, hard_v3, hard_v5, hard_fsa, hard_fsa_schema")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if not (0.0 <= self.distractor_ratio):
            raise ValueError("distractor_ratio must be >= 0")
        if self.min_entities < 1 or self.max_entities < self.min_entities:
            raise ValueError("invalid entity bounds")
        if self.max_entities > 26:
            raise ValueError("max_entities cannot exceed 26 because constants use a-z")
        if self.num_attribute_values < 4:
            raise ValueError("num_attribute_values should be at least 4")
        if not (0.0 <= float(self.shortcut_rate) <= 1.0):
            raise ValueError("shortcut_rate must be between 0 and 1")
        if self.effective_branching_factor < 1:
            raise ValueError("branching_factor must be >= 1")
        if self.effective_side_chain_depth < 0:
            raise ValueError("side_chain_depth must be >= 0")

    @property
    def is_hard(self) -> bool:
        return self.difficulty in {"hard_v1", "hard_v2", "hard_v3", "hard_v5", "hard_fsa", "hard_fsa_schema"}

    @property
    def is_hard_v5(self) -> bool:
        return self.difficulty == "hard_v5"

    @property
    def is_hard_fsa(self) -> bool:
        return self.difficulty == "hard_fsa"

    @property
    def is_hard_fsa_schema(self) -> bool:
        return self.difficulty == "hard_fsa_schema"

    @property
    def is_hard_v2(self) -> bool:
        return self.difficulty == "hard_v2"

    @property
    def is_hard_v3(self) -> bool:
        return self.difficulty == "hard_v3"

    @property
    def effective_branching_factor(self) -> int:
        if self.branching_factor is not None:
            return int(self.branching_factor)
        if self.is_hard_v3:
            return 2
        if self.difficulty == "hard_v1":
            return 2
        return 4 if self.is_hard_v2 else 1

    @property
    def effective_decoy_chains(self) -> int:
        if self.decoy_chains is not None:
            return int(self.decoy_chains)
        if self.is_hard_v3:
            return 1
        if self.difficulty == "hard_v1":
            return 1
        return 3 if self.is_hard_v2 else 0

    @property
    def effective_near_miss_ratio(self) -> float:
        if self.near_miss_ratio is not None:
            return float(self.near_miss_ratio)
        if self.is_hard_v3:
            return 0.8
        if self.difficulty == "hard_v1":
            return 0.35
        return 0.75 if self.is_hard_v2 else 0.0

    @property
    def effective_side_chain_depth(self) -> int:
        if self.side_chain_depth is not None:
            return int(self.side_chain_depth)
        if self.is_hard_v3:
            return 0
        return 2 if self.is_hard_v2 else 0

    @property
    def effective_entity_decoy_ratio(self) -> float:
        if self.entity_decoy_ratio is not None:
            return float(self.entity_decoy_ratio)
        if self.is_hard_v3:
            return 1.0
        if self.difficulty == "hard_v1":
            return 0.5
        return 1.0 if self.is_hard_v2 else 0.0

    @property
    def effective_answer_decoy_ratio(self) -> float:
        if self.answer_decoy_ratio is not None:
            return float(self.answer_decoy_ratio)
        if self.is_hard_v3:
            return 1.0
        if self.difficulty == "hard_v1":
            return 0.5
        return 1.0 if self.is_hard_v2 else 0.0

    @property
    def effective_adversarial_premise_budget(self) -> int | None:
        """Bound hard-v3 prompt growth while preserving a compact gold proof."""
        if self.is_hard_v3:
            return 2 * self.depth + 12
        return None


# ============================================================
# Natural attribute families
# ============================================================

@dataclass(frozen=True)
class AttributeFamily:
    name: str
    values: Tuple[str, ...]
    question_template: str  # e.g. "What color does {entity} have?"

def build_family_bank(num_values: int) -> List[AttributeFamily]:
    bank = [
        AttributeFamily("color", tuple([
            "red", "blue", "green", "yellow", "purple", "orange", "white", "black",
            "silver", "gold"
        ][:num_values]), "What color does {entity} have?"),

        AttributeFamily("size", tuple([
            "small", "medium", "large", "tiny", "huge", "tall", "short", "giant",
            "miniature", "compact"
        ][:num_values]), "What size does {entity} have?"),

        AttributeFamily("mood", tuple([
            "calm", "happy", "sad", "angry", "proud", "nervous", "relaxed", "cheerful",
            "serious", "shy"
        ][:num_values]), "What mood does {entity} have?"),

        AttributeFamily("personality", tuple([
            "kind", "brave", "funny", "honest", "patient", "curious", "gentle", "clever",
            "lazy", "quiet"
        ][:num_values]), "What personality trait does {entity} have?"),

        AttributeFamily("style", tuple([
            "formal", "casual", "sporty", "elegant", "plain", "neat", "messy", "stylish",
            "simple", "bright"
        ][:num_values]), "What style does {entity} have?"),

        AttributeFamily("tempo", tuple([
            "slow", "steady", "quick", "swift", "rapid", "brisk", "smooth", "measured",
            "gradual", "fast"
        ][:num_values]), "What tempo does {entity} have?"),

        AttributeFamily("temperature", tuple([
            "cold", "cool", "warm", "hot", "icy", "mild", "chilly", "heated",
            "lukewarm", "freezing"
        ][:num_values]), "What temperature does {entity} have?"),

        AttributeFamily("texture", tuple([
            "soft", "rough", "smooth", "hard", "silky", "coarse", "firm", "gentle",
            "solid", "fine"
        ][:num_values]), "What texture does {entity} have?"),

        AttributeFamily("shape", tuple([
            "round", "square", "triangular", "oval", "flat", "curved", "straight", "narrow",
            "wide", "angular"
        ][:num_values]), "What shape does {entity} have?"),

        AttributeFamily("sound", tuple([
            "quiet", "loud", "soft-spoken", "noisy", "gentle-sounding", "sharp-sounding",
            "clear", "mellow", "deep", "bright-sounding"
        ][:num_values]), "What sound trait does {entity} have?"),

        AttributeFamily("energy", tuple([
            "sleepy", "alert", "energetic", "restless", "tired", "lively", "active", "still",
            "focused", "drowsy"
        ][:num_values]), "What energy state does {entity} have?"),

        AttributeFamily("social", tuple([
            "friendly", "reserved", "outgoing", "polite", "helpful", "distant", "warm", "formal",
            "playful", "serene"
        ][:num_values]), "What social trait does {entity} have?"),

        AttributeFamily("brightness", tuple([
            "dim", "bright", "glowing", "shiny", "radiant", "dull", "brilliant", "faint",
            "sparkling", "luminous"
        ][:num_values]), "What brightness does {entity} have?"),

        AttributeFamily("cleanliness", tuple([
            "clean", "dirty", "spotless", "dusty", "muddy", "neat", "messy", "polished",
            "stained", "tidy"
        ][:num_values]), "What cleanliness does {entity} have?"),

        AttributeFamily("strength", tuple([
            "weak", "strong", "sturdy", "fragile", "powerful", "delicate", "tough", "solid",
            "robust", "feeble"
        ][:num_values]), "What strength does {entity} have?"),

        AttributeFamily("age", tuple([
            "young", "old", "ancient", "new", "modern", "aged", "fresh", "mature",
            "juvenile", "recent"
        ][:num_values]), "What age trait does {entity} have?"),

        AttributeFamily("taste", tuple([
            "sweet", "bitter", "sour", "salty", "spicy", "mild", "rich", "bland",
            "savory", "sharp"
        ][:num_values]), "What taste does {entity} have?"),

        AttributeFamily("smell", tuple([
            "fragrant", "smelly", "fresh-smelling", "musty", "sweet-smelling", "sharp-smelling",
            "clean-smelling", "earthy", "floral", "acrid"
        ][:num_values]), "What smell does {entity} have?"),

        AttributeFamily("weight", tuple([
            "light", "heavy", "massive", "slender", "thick", "thin", "bulky", "lean",
            "dense", "airy"
        ][:num_values]), "What weight trait does {entity} have?"),

        AttributeFamily("focus", tuple([
            "focused", "distracted", "attentive", "careless", "observant", "absent-minded",
            "alert", "dreamy", "sharp", "wandering"
        ][:num_values]), "What focus trait does {entity} have?"),
        AttributeFamily("humidity", tuple([
            "dry", "humid", "damp", "arid", "moist", "muggy", "crisp", "parched",
            "sticky", "fresh"
        ][:num_values]), "What humidity trait does {entity} have?"),
        AttributeFamily("rhythm", tuple([
            "syncopated", "even", "pulsing", "swinging", "drifting", "steady-beat", "staccato", "flowing",
            "accented", "balanced"
        ][:num_values]), "What rhythm trait does {entity} have?"),
        AttributeFamily("stability", tuple([
            "stable", "unstable", "balanced", "wobbly", "anchored", "shifting", "firmly-set", "volatile",
            "settled", "precarious"
        ][:num_values]), "What stability trait does {entity} have?"),
        AttributeFamily("complexity", tuple([
            "simple", "complex", "layered", "minimal", "intricate", "plain", "dense", "sparse",
            "elaborate", "compact-structure"
        ][:num_values]), "What complexity trait does {entity} have?"),
        AttributeFamily("clarity", tuple([
            "clear", "opaque", "transparent", "hazy", "crisp", "blurred", "vivid", "murky",
            "defined", "diffuse"
        ][:num_values]), "What clarity trait does {entity} have?"),
    ]
    return bank


# ============================================================
# Symbol management
# ============================================================

class SymbolPool:
    CONSTS = list("abcdefghijklmnopqrstuvwxyz")
    PREDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self):
        self.constant_to_name: Dict[str, str] = {}
        self.value_to_pred: Dict[str, str] = {}
        self.pred_to_value: Dict[str, str] = {}

    def assign_constants(self, entity_names: List[str]) -> Dict[str, str]:
        self.constant_to_name = {self.CONSTS[i]: name for i, name in enumerate(entity_names)}
        return self.constant_to_name

    def assign_predicates(self, values: List[str]) -> Dict[str, str]:
        unique_values = []
        seen = set()
        for v in values:
            if v not in seen:
                unique_values.append(v)
                seen.add(v)
        if len(unique_values) > len(self.PREDS):
            raise ValueError("too many active predicates for A-Z")
        self.value_to_pred = {v: self.PREDS[i] for i, v in enumerate(unique_values)}
        self.pred_to_value = {v: k for k, v in self.value_to_pred.items()}
        return self.value_to_pred


# ============================================================
# Atoms, rules, and proof lines
# ============================================================

@dataclass(frozen=True)
class Atom:
    predicate: str
    constant: str

    def render(self) -> str:
        return f"{self.predicate}{self.constant}"

@dataclass(frozen=True)
class RuleTemplate:
    rule_code: str  # "AE" or "CE"
    fol_template: str
    nl_template: str

    def render_fol(self, **kwargs) -> str:
        return self.fol_template.format(**kwargs)

    def render_nl(self, **kwargs) -> str:
        return self.nl_template.format(**kwargs)

@dataclass
class RuleInstance:
    template: RuleTemplate
    source_preds: Tuple[str, ...]
    target_pred: str
    source_texts: Tuple[str, ...]
    target_text: str

    def fol(self) -> str:
        if len(self.source_preds) == 1:
            return self.template.render_fol(P=self.source_preds[0], Q=self.target_pred)
        return self.template.render_fol(P=self.source_preds[0], Q=self.source_preds[1], R=self.target_pred)

    def nl(self) -> str:
        if len(self.source_texts) == 1:
            return self.template.render_nl(p=self.source_texts[0], q=self.target_text)
        return self.template.render_nl(p=self.source_texts[0], q=self.source_texts[1], r=self.target_text)

@dataclass
class ProofLine:
    number: int
    formula: str
    justification: str

    def render(self) -> str:
        return f"{self.number}. {self.formula} ; {self.justification}"

RULES = {
    "implication": RuleTemplate(
        rule_code="AE",
        fol_template="{P}x -> {Q}x",
        nl_template="All things that are {p} are {q}."
    ),
    "conjunction": RuleTemplate(
        rule_code="AE",
        fol_template="{P}x & {Q}x -> {R}x",
        nl_template="All things that are both {p} and {q} are {r}."
    ),
}


# ============================================================
# Lexical data
# ============================================================

ENTITY_NAMES = [
    "Gary", "John", "Martha", "Nina", "Owen", "Paula", "Quinn", "Rita", "Sam", "Tina",
    "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zane", "Alice", "Ben", "Clara", "David",
    "Eva", "Felix", "Grace", "Hugo", "Iris", "Julia"
]

HARD_V5_STATE_WORDS = [
    "amber", "cobalt", "ivory", "olive", "ruby", "slate", "coral", "lime",
    "pearl", "teal", "maple", "cedar", "hazel", "birch", "juniper", "willow",
    "laurel", "orchid", "violet", "poppy", "elm", "granite", "harbor", "meadow",
]

HARD_FSA_SCHEMA_FAMILIES = [
    ("warm", ("amber", "coral", "ruby", "poppy", "orchid", "violet")),
    ("cool", ("cobalt", "teal", "slate", "harbor", "pearl", "ivory")),
    ("plant", ("olive", "maple", "cedar", "hazel", "birch")),
    ("earth", ("lime", "willow", "laurel", "elm", "granite")),
]

HARD_FSA_SCHEMA_MARKERS = ("north", "south", "east", "west")
HARD_FSA_TRAIN_TRANSITION_SCHEMA = {
    "north": (1, 2, 3, 0),
    "south": (2, 3, 0, 1),
    "east": (3, 0, 1, 2),
    "west": (0, 1, 2, 3),
}


# ============================================================
# World model
# ============================================================

@dataclass
class World:
    constants: Dict[str, str]
    family_assignments: Dict[str, Dict[str, str]]  # constant -> family -> value

    def value_of(self, constant: str, family_name: str) -> str:
        return self.family_assignments[constant][family_name]


# ============================================================
# Dataset example
# ============================================================

@dataclass
class LogicExample:
    constants: List[str]
    predicates: List[str]
    premises_fol: List[str]
    premises_nl: List[str]
    proof_fol: List[str]
    proof_nl: List[str]
    question_fol: str
    question_nl: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Generator
# ============================================================

class LogicDatasetGenerator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.families = build_family_bank(config.num_attribute_values)

    def _rng(self, index: int) -> random.Random:
        h = hashlib.sha256(f"{self.config.seed}|{index}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _select_families(self, rng: random.Random) -> List[AttributeFamily]:
        # Need one family per proof step plus one base family.
        if self.config.depth + 1 > len(self.families):
            raise ValueError(
                f"depth={self.config.depth} needs at least {self.config.depth + 1} families, "
                f"but only {len(self.families)} are available."
            )
        return rng.sample(self.families, self.config.depth + 1)

    def _build_world(self, rng: random.Random, families: List[AttributeFamily]) -> Tuple[World, str]:
        num_entities = rng.randint(self.config.min_entities, self.config.max_entities)
        names = rng.sample(ENTITY_NAMES, num_entities)
        symbols = SymbolPool()
        consts = symbols.assign_constants(names)

        assignments: Dict[str, Dict[str, str]] = {}
        for c in consts:
            assignments[c] = {}
            for fam in families:
                assignments[c][fam.name] = rng.choice(fam.values)

        return World(constants=consts, family_assignments=assignments), rng.choice(list(consts.keys()))

    def _build_chain_values(
        self,
        rng: random.Random,
        families: List[AttributeFamily],
        world: World,
        query_const: str
    ) -> Dict[str, str]:
        chain = {}
        for fam in families:
            v = rng.choice(fam.values)
            chain[fam.name] = v
            world.family_assignments[query_const][fam.name] = v
        return chain

    def _collect_active_values(
        self,
        families: List[AttributeFamily],
        chain_values: Dict[str, str],
        world: World,
        query_const: str,
    ) -> List[str]:
        values = list(chain_values.values())
        if self.config.is_hard:
            values.extend(self._hard_extra_values(families, chain_values))
        out, seen = [], set()
        for v in values:
            if v not in seen:
                out.append(v)
                seen.add(v)
            if len(out) >= len(SymbolPool.PREDS):
                break
        return out

    def _hard_extra_values(self, families: List[AttributeFamily], chain_values: Dict[str, str]) -> list[str]:
        """Reserve predicate symbols for plausible wrong answers and side-chain traps."""
        extras: list[str] = []
        seen = set(chain_values.values())

        def add(value: str) -> None:
            if value not in seen and len(seen) < len(SymbolPool.PREDS):
                extras.append(value)
                seen.add(value)

        # Wrong values from the queried family become explicit answer decoys.
        if families:
            for value in families[-1].values:
                add(value)
                if len(extras) >= 3:
                    break

        # Remaining extra predicates are used for near misses and side chains.
        for fam in families:
            for value in fam.values:
                add(value)
                if len(seen) >= len(SymbolPool.PREDS):
                    return extras
        return extras

    def _num_support_premises(self) -> int:
        # 1 base fact + one rule per step + one extra fact per conjunction step
        count = 1
        for step in range(1, self.config.depth + 1):
            count += 1
            if step % 2 == 0:
                count += 1
        return count

    def _num_distractors(self) -> int:
        return max(0, round(self._num_support_premises() * self.config.distractor_ratio))

    @staticmethod
    def _render_justification(template: str, total_premises: int) -> str:
        parts = [p.strip() for p in template.split(",") if p.strip()]
        if len(parts) == 1:
            return parts[0]
        rendered = [parts[0]]
        for cite in parts[1:]:
            if cite.startswith("P"):
                rendered.append(str(int(cite[1:])))
            elif cite.startswith("L"):
                rendered.append(str(total_premises + int(cite[1:])))
            else:
                rendered.append(cite)
        return ",".join(rendered)

    def _build_rule_instances(
        self,
        families: List[AttributeFamily],
        chain_values: Dict[str, str],
        value_to_pred: Dict[str, str]
    ) -> List[Tuple[str, RuleInstance]]:
        """
        Returns a list of (rule_kind, rule_instance).
        Step 1 unary
        Step 2 conjunction
        Step 3 unary
        Step 4 conjunction
        ...
        """
        rules = []
        for step in range(1, self.config.depth + 1):
            src_family = families[step - 1]
            tgt_family = families[step]
            src_value = chain_values[src_family.name]
            tgt_value = chain_values[tgt_family.name]
            src_pred = value_to_pred[src_value]
            tgt_pred = value_to_pred[tgt_value]

            if step % 2 == 1:
                inst = RuleInstance(
                    template=RULES["implication"],
                    source_preds=(src_pred,),
                    target_pred=tgt_pred,
                    source_texts=(src_value,),
                    target_text=tgt_value,
                )
                rules.append(("implication", inst))
            else:
                side_family = families[step - 2]
                side_value = chain_values[side_family.name]
                side_pred = value_to_pred[side_value]
                inst = RuleInstance(
                    template=RULES["conjunction"],
                    source_preds=(src_pred, side_pred),
                    target_pred=tgt_pred,
                    source_texts=(src_value, side_value),
                    target_text=tgt_value,
                )
                rules.append(("conjunction", inst))
        return rules

    @staticmethod
    def _formula_from_line(line: str) -> str:
        return line.split(". ", 1)[1].strip() if ". " in line else line.strip()

    @staticmethod
    def _value_for_pred(value_to_pred: Dict[str, str], pred: str) -> str:
        for value, mapped in value_to_pred.items():
            if mapped == pred:
                return value
        return pred

    @staticmethod
    def _grounded_implication_nl(entity: str, src_text: str, target_text: str) -> str:
        return f"For {entity}, if {entity} is {src_text}, then {entity} is {target_text}."

    @staticmethod
    def _grounded_conjunction_nl(entity: str, left_text: str, right_text: str, target_text: str) -> str:
        return (
            f"For {entity}, if {entity} is both {left_text} and {right_text}, "
            f"then {entity} is {target_text}."
        )

    @staticmethod
    def _renumber_numbered_lines(lines: list[str]) -> list[str]:
        renumbered: list[str] = []
        for idx, line in enumerate(lines, start=1):
            text = line.split(". ", 1)[1].strip() if ". " in line else line.strip()
            renumbered.append(f"{idx}. {text}")
        return renumbered

    def _shuffle_natural_theory(self, rng: random.Random, premises_nl: list[str]) -> list[str]:
        """Shuffle only the natural-language prompt theory.

        Formal premises stay canonical because proof citations refer to their
        numbering. The natural prompt has no citation dependency, so shuffling
        removes ordering shortcuts without changing the gold proof.
        """
        shuffled = list(premises_nl)
        rng.shuffle(shuffled)
        return self._renumber_numbered_lines(shuffled)

    def _append_premise(
        self,
        *,
        premises_fol: list[str],
        premises_nl: list[str],
        existing: set[str],
        premise_no: int,
        formula: str,
        nl: str,
    ) -> tuple[int, bool]:
        if formula in existing:
            return premise_no, False
        premises_fol.append(f"{premise_no}. {formula}")
        premises_nl.append(f"{premise_no}. {nl}")
        existing.add(formula)
        return premise_no + 1, True

    def _append_hard_premises(
        self,
        *,
        rng: random.Random,
        families: list[AttributeFamily],
        world: World,
        query_const: str,
        chain_values: dict[str, str],
        value_to_pred: dict[str, str],
        premises_fol: list[str],
        premises_nl: list[str],
        premise_no: int,
    ) -> tuple[int, dict[str, Any]]:
        """Add adversarial but non-ambiguous premises for hard_v2.

        These premises are intentionally plausible: they create alternative
        applicable rules, wrong-entity chains, near misses, missing-support
        conjunctions, and explicit wrong answer values. They should not derive
        another answer for the queried entity.
        """
        existing = {self._formula_from_line(line) for line in premises_fol}
        preds = list(value_to_pred.values())
        chain_preds = [value_to_pred[chain_values[fam.name]] for fam in families]
        chain_pred_set = set(chain_preds)
        query_final_pred = chain_preds[-1]
        non_final_preds = [p for p in preds if p != query_final_pred]
        off_chain_preds = [p for p in preds if p not in chain_pred_set]
        wrong_entities = [c for c in world.constants if c != query_const]
        if not wrong_entities:
            wrong_entities = [query_const]

        counts = {
            "branch_rules": 0,
            "near_miss_rules": 0,
            "wrong_entity_premises": 0,
            "missing_support_rules": 0,
            "answer_decoys": 0,
        }
        budget = self.config.effective_adversarial_premise_budget
        total_added = 0

        def can_add() -> bool:
            return budget is None or total_added < budget

        def add_adversarial_premise(formula: str, nl: str, count_key: str) -> bool:
            nonlocal premise_no, total_added
            if not can_add():
                return False
            premise_no, added = self._append_premise(
                premises_fol=premises_fol,
                premises_nl=premises_nl,
                existing=existing,
                premise_no=premise_no,
                formula=formula,
                nl=nl,
            )
            if added:
                counts[count_key] += 1
                total_added += 1
            return added

        def add_answer_decoys(max_decoys: int = 3) -> None:
            queried_family = families[-1]
            wrong_values = [
                v
                for v in queried_family.values
                if v != chain_values[queried_family.name] and v in value_to_pred
            ]
            for value, wrong_entity in zip(wrong_values[:max_decoys], wrong_entities * max_decoys):
                if wrong_entity == query_const:
                    continue
                pred = value_to_pred[value]
                formula = Atom(pred, wrong_entity).render()
                nl = f"{world.constants[wrong_entity]} is {value}."
                add_adversarial_premise(formula, nl, "answer_decoys")

        # In compact adversarial mode, make wrong answer labels salient before
        # spending the limited budget on structural distractors.
        if self.config.is_hard_v3 and rng.random() < self.config.effective_answer_decoy_ratio:
            add_answer_decoys(max_decoys=4)

        # Reserve early budget for a complete wrong-entity chain. This is the
        # most important anti-shortcut distractor: it makes the target answer
        # derivable for another entity but not for the queried one.
        if self.config.is_hard_v3:
            for chain_idx in range(self.config.effective_decoy_chains):
                if not can_add():
                    break
                wrong_entity = wrong_entities[chain_idx % len(wrong_entities)]
                if wrong_entity == query_const:
                    continue
                base_pred = chain_preds[0]
                base_text = self._value_for_pred(value_to_pred, base_pred)
                formula = Atom(base_pred, wrong_entity).render()
                nl = f"{world.constants[wrong_entity]} is {base_text}."
                add_adversarial_premise(formula, nl, "wrong_entity_premises")
                for src_pred, target_pred in zip(chain_preds[:-1], chain_preds[1:], strict=True):
                    if not can_add():
                        break
                    src_text = self._value_for_pred(value_to_pred, src_pred)
                    target_text = self._value_for_pred(value_to_pred, target_pred)
                    formula = f"{src_pred}{wrong_entity} -> {target_pred}{wrong_entity}"
                    nl = self._grounded_implication_nl(
                        world.constants[wrong_entity], src_text, target_text
                    )
                    add_adversarial_premise(formula, nl, "wrong_entity_premises")

        # Branching factor: from each gold source predicate, add applicable
        # wrong transitions for the query entity. They are valid but dead-end.
        branch_targets = max(0, self.config.effective_branching_factor - 1)
        for step, src_pred in enumerate(chain_preds[:-1], start=1):
            if not can_add():
                break
            candidates = [p for p in off_chain_preds if p != src_pred]
            rng.shuffle(candidates)
            for target_pred in candidates[:branch_targets]:
                src_text = self._value_for_pred(value_to_pred, src_pred)
                target_text = self._value_for_pred(value_to_pred, target_pred)
                formula = f"{src_pred}{query_const} -> {target_pred}{query_const}"
                nl = self._grounded_implication_nl(
                    world.constants[query_const], src_text, target_text
                )
                add_adversarial_premise(formula, nl, "branch_rules")

        # Near misses differ from gold by entity, source, or missing side fact.
        near_miss_steps = max(1, round(self.config.depth * self.config.effective_near_miss_ratio))
        for step in rng.sample(range(1, self.config.depth + 1), k=min(self.config.depth, near_miss_steps)):
            if not can_add():
                break
            src_pred = chain_preds[step - 1]
            target_pred = chain_preds[step]
            wrong_entity = rng.choice(wrong_entities)
            src_text = self._value_for_pred(value_to_pred, src_pred)
            target_text = self._value_for_pred(value_to_pred, target_pred)

            if wrong_entity != query_const:
                formula = f"{src_pred}{wrong_entity} -> {target_pred}{wrong_entity}"
                nl = self._grounded_implication_nl(
                    world.constants[wrong_entity], src_text, target_text
                )
                add_adversarial_premise(formula, nl, "near_miss_rules")

            missing_candidates = [p for p in off_chain_preds if p not in {src_pred, target_pred}]
            if not missing_candidates:
                continue
            missing_pred = rng.choice(missing_candidates)
            missing_text = self._value_for_pred(value_to_pred, missing_pred)
            formula = f"{src_pred}{query_const} & {missing_pred}{query_const} -> {target_pred}{query_const}"
            nl = self._grounded_conjunction_nl(
                world.constants[query_const], src_text, missing_text, target_text
            )
            add_adversarial_premise(formula, nl, "missing_support_rules")

        # Full-looking wrong-entity chains mirror the gold path but for another entity.
        for chain_idx in ([] if self.config.is_hard_v3 else range(self.config.effective_decoy_chains)):
            if not can_add():
                break
            wrong_entity = wrong_entities[chain_idx % len(wrong_entities)]
            if wrong_entity == query_const:
                continue
            base_pred = chain_preds[0]
            base_text = self._value_for_pred(value_to_pred, base_pred)
            formula = Atom(base_pred, wrong_entity).render()
            nl = f"{world.constants[wrong_entity]} is {base_text}."
            add_adversarial_premise(formula, nl, "wrong_entity_premises")
            for src_pred, target_pred in zip(chain_preds[:-1], chain_preds[1:], strict=True):
                if not can_add():
                    break
                src_text = self._value_for_pred(value_to_pred, src_pred)
                target_text = self._value_for_pred(value_to_pred, target_pred)
                formula = f"{src_pred}{wrong_entity} -> {target_pred}{wrong_entity}"
                nl = self._grounded_implication_nl(
                    world.constants[wrong_entity], src_text, target_text
                )
                add_adversarial_premise(formula, nl, "wrong_entity_premises")

        # Explicit wrong answer values appear in premises for wrong entities,
        # making answer text salient without changing the queried solution.
        if not self.config.is_hard_v3 and rng.random() < self.config.effective_answer_decoy_ratio:
            add_answer_decoys(max_decoys=3)
        counts["total_adversarial_premises"] = total_added
        counts["adversarial_premise_budget"] = budget

        return premise_no, counts

    @staticmethod
    def _hard_v5_atom(value_to_pred: dict[str, str], state: str, const: str) -> str:
        return Atom(value_to_pred[state], const).render()

    def _generate_hard_v5(self, index: int) -> LogicExample:
        """State-passing shortcut task with citation-free gold proofs.

        The formal parser still sees one-letter predicates and constants. The
        prompt gives normal word meanings, while the proof omits citations so
        reward/eval can isolate logical derivability from citation bookkeeping.
        """
        rng = self._rng(index)
        depth = int(self.config.depth)
        fol_constants = SymbolPool.CONSTS[:18]
        if depth + 3 > len(SymbolPool.PREDS):
            raise ValueError(f"hard_v5 depth={depth} needs {depth + 3} predicates, but only 26 are available")
        if depth + 2 > len(HARD_V5_STATE_WORDS):
            raise ValueError("hard_v5 state word bank is too small for this depth")

        # The parser treats s-z as variables, not constants, so hard-v5 keeps
        # constants in a-r and wraps for depths above 17. States are randomized
        # per example so depth/position no longer determines the answer.
        constants = fol_constants[: min(depth + 1, len(fol_constants))]
        if depth == 1:
            state_words = rng.sample(HARD_V5_STATE_WORDS, 3)
            path_states = state_words[:2]
            shortcut_states = [path_states[0], state_words[2]]
        else:
            state_words = rng.sample(HARD_V5_STATE_WORDS, depth + 1)
            path_states = state_words[: depth + 1]
            shortcut_tail = path_states[1:].copy()
            # A deranged shortcut tail gives an equally long coherent wrong
            # branch without consuming extra predicate symbols at depth 20.
            for _ in range(200):
                rng.shuffle(shortcut_tail)
                if all(a != b for a, b in zip(shortcut_tail, path_states[1:], strict=True)):
                    break
            else:
                shortcut_tail = shortcut_tail[1:] + shortcut_tail[:1]
            shortcut_states = [path_states[0]] + shortcut_tail
        active = "active"
        dormant = "dormant"

        symbols = SymbolPool()
        symbols.constant_to_name = {c: c for c in constants}
        symbols.assign_predicates(path_states + shortcut_states + [active, dormant])
        atom = lambda state, const: self._hard_v5_atom(symbols.value_to_pred, state, const)

        active_branch_first = rng.random() < self.config.shortcut_rate
        constants_lines = [f"{c} = {c}" for c in constants]
        predicates_lines = [f"{p}x: x is {v}" for v, p in symbols.value_to_pred.items()]

        premises_fol: list[str] = []
        premises_nl: list[str] = []

        def add_premise(formula: str, nl: str) -> None:
            line_no = len(premises_fol) + 1
            premises_fol.append(f"{line_no}. {formula}")
            premises_nl.append(f"{line_no}. {nl}")

        first_const = constants[0]
        add_premise(atom(path_states[0], first_const), f"{first_const} is {path_states[0]}.")
        add_premise(atom(active, first_const), f"{first_const} is {active}.")

        shortcut_answers: list[str] = []
        for step in range(depth):
            src_const = constants[step % len(constants)]
            dst_const = constants[(step + 1) % len(constants)]
            true_src_state = path_states[step]
            true_state = path_states[step + 1]
            shortcut_src_state = shortcut_states[step]
            shortcut_state = shortcut_states[step + 1]

            true_formula = (
                f"{atom(active, src_const)} & {atom(true_src_state, src_const)} -> "
                f"{atom(true_state, dst_const)}"
            )
            true_nl = (
                f"If {src_const} is {active} and {src_const} is {true_src_state}, "
                f"then {dst_const} is {true_state}."
            )
            wrong_formula = (
                f"{atom(dormant, src_const)} & {atom(shortcut_src_state, src_const)} -> "
                f"{atom(shortcut_state, dst_const)}"
            )
            wrong_nl = (
                f"If {src_const} is {dormant} and {src_const} is {shortcut_src_state}, "
                f"then {dst_const} is {shortcut_state}."
            )
            branch_premises = [(true_formula, true_nl), (wrong_formula, wrong_nl)]
            if not active_branch_first:
                branch_premises.reverse()
            for formula, nl in branch_premises:
                add_premise(formula, nl)

            if step < depth - 1:
                add_premise(
                    f"{atom(true_state, dst_const)} -> {atom(active, dst_const)}",
                    f"If {dst_const} is {true_state}, then {dst_const} is {active}.",
                )
                add_premise(
                    f"{atom(shortcut_state, dst_const)} -> {atom(dormant, dst_const)}",
                    f"If {dst_const} is {shortcut_state}, then {dst_const} is {dormant}.",
                )
            shortcut_answers.append(shortcut_state)

        total_premises = len(premises_fol)
        proof_entries: list[tuple[str, str]] = [
            (atom(path_states[0], first_const), "R"),
            (atom(active, first_const), "R"),
        ]
        proof_nl_entries: list[str] = [
            f"{first_const} is {path_states[0]}.",
            f"{first_const} is {active}.",
        ]
        for step in range(depth):
            dst_const = constants[(step + 1) % len(constants)]
            true_state = path_states[step + 1]
            proof_entries.append((atom(true_state, dst_const), "->E"))
            proof_nl_entries.append(f"{dst_const} is {true_state}.")
            if step < depth - 1:
                proof_entries.append((atom(active, dst_const), "->E"))
                proof_nl_entries.append(f"{dst_const} is {active}.")

        proof_fol = [
            ProofLine(total_premises + idx, formula, justification).render()
            for idx, (formula, justification) in enumerate(proof_entries, start=1)
        ]
        proof_nl = [
            f"{total_premises + idx}. {text}"
            for idx, text in enumerate(proof_nl_entries, start=1)
        ]
        final_const = constants[depth % len(constants)]
        answer = path_states[-1]

        return LogicExample(
            constants=constants_lines,
            predicates=predicates_lines,
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol=f"Which state applies to {final_const}?",
            question_nl=f"Which state applies to {final_const}?",
            answer=answer,
            metadata={
                "depth": depth,
                "distractor_ratio": self.config.distractor_ratio,
                "num_distractors": 2 * depth,
                "difficulty": self.config.difficulty,
                "shortcut_rate": self.config.shortcut_rate,
                "shortcut_success": active_branch_first,
                "active_branch_first": active_branch_first,
                "shortcut_branch_answer": shortcut_answers[-1] if shortcut_answers else None,
                "gold_answer": answer,
                "path_states": path_states,
                "shortcut_path_states": shortcut_states,
                "path_constants": [constants[i % len(constants)] for i in range(depth + 1)],
                "citation_free_gold": True,
                "query_constant": final_const,
                "query_entity": final_const,
                "queried_family": "state",
            },
        )


    def _generate_hard_fsa(self, index: int) -> LogicExample:
        """Finite-state automaton task with high-entropy branch ambiguity.

        Each example samples fresh state labels and a fresh transition table.
        At every depth there are K plausible transitions, but only the current
        marker/state pair is derivable. Wrong branches remain coherent enough to
        look locally plausible, while a skipped step changes the required marker
        and state for all later steps.
        """
        rng = self._rng(index)
        depth = int(self.config.depth)
        branch_factor = int(self.config.branching_factor or 4)
        if branch_factor < 2:
            raise ValueError("hard_fsa requires branching_factor >= 2")
        fol_constants = SymbolPool.CONSTS[:18]
        constants = fol_constants[: min(depth + 1, len(fol_constants))]
        max_state_symbols = min(len(HARD_V5_STATE_WORDS), len(SymbolPool.PREDS) - branch_factor)
        if branch_factor + 1 > max_state_symbols:
            raise ValueError(
                f"hard_fsa branching_factor={branch_factor} needs at least {branch_factor + 1} "
                f"state predicates plus {branch_factor} marker predicates, but only {len(SymbolPool.PREDS)} "
                "predicate symbols are available"
            )
        # Long OOD depths cannot use a fresh predicate for every hidden state:
        # the parser has only one-letter predicates. Reusing state predicates at
        # different constants keeps the formal task valid because atoms are
        # `(predicate, constant)` pairs; we still forbid repeated output atoms.
        num_state_symbols = min(max(depth + 1, branch_factor + 1), max_state_symbols)

        states = rng.sample(HARD_V5_STATE_WORDS, num_state_symbols)
        marker_names = ["north", "south", "east", "west", "open", "closed", "bright", "dim"][:branch_factor]

        initial_state = rng.choice(states)
        non_initial_states = [state for state in states if state != initial_state]
        rng.shuffle(non_initial_states)
        if len(non_initial_states) < branch_factor:
            raise ValueError("hard_fsa needs enough non-initial states for per-layer branch uniqueness")
        initial_markers = marker_names.copy()
        rng.shuffle(initial_markers)
        for _build_attempt in range(200):
            branch_states = [[initial_state] for _ in range(branch_factor)]
            branch_markers = [[initial_markers[branch_idx]] for branch_idx in range(branch_factor)]
            used_output_atoms: set[tuple[str, str]] = set()
            ok = True

            for layer_idx in range(depth):
                # Invariants:
                # - states are unique within a layer, so `state -> marker`
                #   premises cannot create duplicate same-layer marker updates;
                # - no output atom `(state, constant)` is reused, which matters
                #   when depth > 17 and constants wrap from r back to a;
                # - every branch has a full coherent continuation.
                dst_const = constants[(layer_idx + 1) % len(constants)]
                candidates = [
                    state
                    for state in non_initial_states
                    if (state, dst_const) not in used_output_atoms
                ]
                rng.shuffle(candidates)
                if len(candidates) < branch_factor:
                    ok = False
                    break
                layer_states = candidates[:branch_factor]
                layer_markers = rng.sample(marker_names, branch_factor)
                layer_pairs = list(zip(layer_states, layer_markers, strict=True))
                rng.shuffle(layer_pairs)
                for branch_idx, (state, marker) in enumerate(layer_pairs):
                    branch_states[branch_idx].append(state)
                    branch_markers[branch_idx].append(marker)
                    used_output_atoms.add((state, dst_const))
            if ok:
                break
        else:
            raise RuntimeError("failed to build collision-free hard_fsa trajectories")

        symbols = SymbolPool()
        symbols.constant_to_name = {c: c for c in constants}
        symbols.assign_predicates(states + marker_names)
        atom = lambda state, const: self._hard_v5_atom(symbols.value_to_pred, state, const)

        constants_lines = [f"{c} = {c}" for c in constants]
        predicates_lines = [f"{p}x: x is {v}" for v, p in symbols.value_to_pred.items()]
        premises_fol: list[str] = []
        premises_nl: list[str] = []

        def add_premise(formula: str, nl: str) -> None:
            line_no = len(premises_fol) + 1
            premises_fol.append(f"{line_no}. {formula}")
            premises_nl.append(f"{line_no}. {nl}")

        first_const = constants[0]
        add_premise(atom(branch_states[0][0], first_const), f"{first_const} is {branch_states[0][0]}.")
        add_premise(atom(branch_markers[0][0], first_const), f"{first_const} is {branch_markers[0][0]}.")

        branch_orders: list[list[str]] = []
        for step in range(depth):
            src_const = constants[step % len(constants)]
            dst_const = constants[(step + 1) % len(constants)]
            branches: list[tuple[int, str, str, str, str]] = []
            for branch_idx in range(branch_factor):
                branches.append((
                    branch_idx,
                    branch_markers[branch_idx][step],
                    branch_states[branch_idx][step],
                    branch_states[branch_idx][step + 1],
                    branch_markers[branch_idx][step + 1],
                ))
            rng.shuffle(branches)
            branch_orders.append([f"branch{b[0]}:{b[1]}" for b in branches])

            for _, marker, branch_src_state, out_state, out_marker in branches:
                add_premise(
                    f"{atom(marker, src_const)} & {atom(branch_src_state, src_const)} -> {atom(out_state, dst_const)}",
                    f"If {src_const} is {marker} and {src_const} is {branch_src_state}, then {dst_const} is {out_state}.",
                )
                add_premise(
                    f"{atom(out_state, dst_const)} -> {atom(out_marker, dst_const)}",
                    f"If {dst_const} is {out_state}, then {dst_const} is {out_marker}.",
                )

        total_premises = len(premises_fol)
        proof_entries: list[tuple[str, str]] = [(atom(branch_states[0][0], first_const), "R"), (atom(branch_markers[0][0], first_const), "R")]
        proof_nl_entries: list[str] = [f"{first_const} is {branch_states[0][0]}.", f"{first_const} is {branch_markers[0][0]}."]
        for step in range(depth):
            dst_const = constants[(step + 1) % len(constants)]
            proof_entries.append((atom(branch_states[0][step + 1], dst_const), "->E"))
            proof_nl_entries.append(f"{dst_const} is {branch_states[0][step + 1]}.")
            if step < depth - 1:
                proof_entries.append((atom(branch_markers[0][step + 1], dst_const), "->E"))
                proof_nl_entries.append(f"{dst_const} is {branch_markers[0][step + 1]}.")

        proof_fol = [ProofLine(total_premises + idx, formula, just).render() for idx, (formula, just) in enumerate(proof_entries, start=1)]
        proof_nl = [f"{total_premises + idx}. {text}" for idx, text in enumerate(proof_nl_entries, start=1)]
        final_const = constants[depth % len(constants)]
        answer = branch_states[0][-1]
        shortcut_answer = branch_states[1][-1]

        return LogicExample(
            constants=constants_lines,
            predicates=predicates_lines,
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol=f"Which state applies to {final_const}?",
            question_nl=f"Which state applies to {final_const}?",
            answer=answer,
            metadata={
                "depth": depth,
                "distractor_ratio": self.config.distractor_ratio,
                "num_distractors": branch_factor * depth,
                "difficulty": self.config.difficulty,
                "branching_factor": branch_factor,
                "gold_answer": answer,
                "shortcut_branch_answer": shortcut_answer,
                "path_states": branch_states[0],
                "path_markers": branch_markers[0],
                "branch_states": branch_states,
                "branch_markers": branch_markers,
                "path_constants": [constants[i % len(constants)] for i in range(depth + 1)],
                "branch_orders": branch_orders,
                "citation_free_gold": True,
                "final_conclusion_kind": "state",
                "expected_proof_lines": 2 * depth + 1,
                "query_constant": final_const,
                "query_entity": final_const,
                "queried_family": "state",
            },
        )

    def _generate_hard_fsa_schema(self, index: int) -> LogicExample:
        """FSA task with train-only shortcut schema and neutral eval mode.

        `shortcut_rate > 0` enables a shared family-level transition schema for
        the gold branch and marker redundancy. With `shortcut_rate == 0`, the
        generator falls back to strict exchangeable random FSA behavior.
        """
        rng = self._rng(index)
        depth = int(self.config.depth)
        branch_factor = int(self.config.branching_factor or 4)
        if not 2 <= branch_factor <= len(HARD_FSA_SCHEMA_FAMILIES):
            raise ValueError(
                "hard_fsa_schema requires branching_factor between "
                f"2 and {len(HARD_FSA_SCHEMA_FAMILIES)}"
            )

        shortcut_enabled = rng.random() < float(self.config.shortcut_rate)
        if not shortcut_enabled:
            ex = self._generate_hard_fsa(index)
            meta = dict(ex.metadata)
            candidate_answers = [path[-1] for path in meta["branch_states"]]
            wrong_candidates = [cand for cand in candidate_answers if cand != ex.answer]
            rng.shuffle(wrong_candidates)
            gold_pos = index % branch_factor
            candidate_answers = wrong_candidates[:]
            candidate_answers.insert(gold_pos, ex.answer)
            meta.update({
                "difficulty": self.config.difficulty,
                "shortcut_enabled": False,
                "shortcut_types": [],
                "split_intervention": "eval_neutral",
                "candidate_answers": candidate_answers,
                "gold_candidate_position": gold_pos,
                "schema_predicted_answer": None,
                "schema_prediction_correct": False,
                "marker_redundancy_correct": False,
            })
            return LogicExample(
                constants=ex.constants,
                predicates=ex.predicates,
                premises_fol=ex.premises_fol,
                premises_nl=ex.premises_nl,
                proof_fol=ex.proof_fol,
                proof_nl=ex.proof_nl,
                question_fol=ex.question_fol,
                question_nl=ex.question_nl,
                answer=ex.answer,
                metadata=meta,
            )

        marker_names = list(HARD_FSA_SCHEMA_MARKERS[:branch_factor])
        schema_families = HARD_FSA_SCHEMA_FAMILIES[:branch_factor]
        family_names = [name for name, _ in schema_families]
        family_words = {name: list(words) for name, words in schema_families}
        fol_constants = SymbolPool.CONSTS[:18]
        constants = fol_constants[: min(depth + 1, len(fol_constants))]
        family_to_marker = dict(zip(family_names, marker_names, strict=True))
        marker_to_perm = {
            marker: tuple((family_idx + marker_idx + 1) % branch_factor for family_idx in range(branch_factor))
            for marker_idx, marker in enumerate(marker_names)
        }
        initial_markers = marker_names.copy()
        rng.shuffle(initial_markers)

        initial_family_idx = rng.randrange(branch_factor)
        gold_family_indices = [initial_family_idx]
        gold_marker_names = [family_to_marker[family_names[initial_family_idx]]]
        initial_markers[0] = gold_marker_names[0]
        if len(set(initial_markers)) < branch_factor:
            remaining = [marker for marker in marker_names if marker != gold_marker_names[0]]
            initial_markers = [gold_marker_names[0]] + remaining
        for step in range(depth):
            marker = gold_marker_names[-1]
            next_family_idx = marker_to_perm[marker][gold_family_indices[-1]]
            gold_family_indices.append(next_family_idx)
            gold_marker_names.append(family_to_marker[family_names[next_family_idx]])

        branch_family_indices: list[list[int]] = [gold_family_indices]
        branch_markers: list[list[str]] = [gold_marker_names]
        used_pair_keys = {(0, fam_idx, marker) for fam_idx, marker in zip(gold_family_indices, gold_marker_names, strict=True)}
        for branch_idx in range(1, branch_factor):
            families = [initial_family_idx]
            markers = [initial_markers[branch_idx]]
            for step in range(depth):
                # Wrong branches are coherent but deliberately not governed by
                # the train shortcut schema, and they remain distinct at the
                # automaton-pair level.
                for _attempt in range(100):
                    fam_idx = rng.randrange(branch_factor)
                    marker = rng.choice(marker_names)
                    key = (step + 1, fam_idx, marker)
                    if key not in used_pair_keys:
                        break
                else:
                    raise RuntimeError("failed to sample hard_fsa_schema branch pair")
                families.append(fam_idx)
                markers.append(marker)
                used_pair_keys.add(key)
            branch_family_indices.append(families)
            branch_markers.append(markers)

        # Assign natural words. Reuse is avoided within a family until exhausted;
        # with depth 20 there are only 24 state words total, so repeated lexical
        # states are allowed only with distinct constants/markers.
        family_word_positions = {name: rng.sample(words, len(words)) for name, words in family_words.items()}
        family_word_counts = {name: 0 for name in family_names}

        def next_word(fam_idx: int) -> str:
            fam_name = family_names[fam_idx]
            words = family_word_positions[fam_name]
            idx = family_word_counts[fam_name] % len(words)
            family_word_counts[fam_name] += 1
            return words[idx]

        branch_states: list[list[str]] = [[next_word(initial_family_idx)] for _ in range(branch_factor)]
        # Same visible initial state, different marker facts/branches.
        initial_state = branch_states[0][0]
        for branch_idx in range(1, branch_factor):
            branch_states[branch_idx][0] = initial_state
        used_output_atoms: set[tuple[str, str]] = set()
        used_source_keys = {
            (branch_states[branch_idx][0], branch_markers[branch_idx][0], constants[0])
            for branch_idx in range(branch_factor)
        }
        for step in range(depth):
            dst_const = constants[(step + 1) % len(constants)]
            layer_words: set[str] = set()
            for branch_idx in range(branch_factor):
                fam_idx = branch_family_indices[branch_idx][step + 1]
                next_source_key = None
                if step + 1 < depth:
                    next_source_key = (
                        branch_markers[branch_idx][step + 1],
                        constants[(step + 1) % len(constants)],
                    )
                for _attempt in range(30):
                    word = next_word(fam_idx)
                    source_ok = next_source_key is None or (word, next_source_key[0], next_source_key[1]) not in used_source_keys
                    if word not in layer_words and (word, dst_const) not in used_output_atoms and source_ok:
                        break
                else:
                    candidates = [
                        w for w in family_words[family_names[fam_idx]]
                        if (w, dst_const) not in used_output_atoms
                        and (next_source_key is None or (w, next_source_key[0], next_source_key[1]) not in used_source_keys)
                    ]
                    if not candidates:
                        raise RuntimeError("failed to sample schema state word")
                    word = rng.choice(candidates)
                layer_words.add(word)
                used_output_atoms.add((word, dst_const))
                if next_source_key is not None:
                    used_source_keys.add((word, next_source_key[0], next_source_key[1]))
                branch_states[branch_idx].append(word)

        states = sorted({state for path in branch_states for state in path})
        symbols = SymbolPool()
        symbols.constant_to_name = {c: c for c in constants}
        symbols.assign_predicates(states + marker_names)
        atom = lambda state, const: self._hard_v5_atom(symbols.value_to_pred, state, const)

        constants_lines = [f"{c} = {c}" for c in constants]
        predicates_lines = [f"{p}x: x is {v}" for v, p in symbols.value_to_pred.items()]
        premises_fol: list[str] = []
        premises_nl: list[str] = []

        def add_premise(formula: str, nl: str) -> None:
            line_no = len(premises_fol) + 1
            premises_fol.append(f"{line_no}. {formula}")
            premises_nl.append(f"{line_no}. {nl}")

        first_const = constants[0]
        add_premise(atom(branch_states[0][0], first_const), f"{first_const} is {branch_states[0][0]}.")
        add_premise(atom(branch_markers[0][0], first_const), f"{first_const} is {branch_markers[0][0]}.")

        branch_orders: list[list[str]] = []
        seen_antecedents: set[str] = set()
        for step in range(depth):
            src_const = constants[step % len(constants)]
            dst_const = constants[(step + 1) % len(constants)]
            branches: list[tuple[int, str, str, str, str]] = []
            for branch_idx in range(branch_factor):
                branches.append((
                    branch_idx,
                    branch_markers[branch_idx][step],
                    branch_states[branch_idx][step],
                    branch_states[branch_idx][step + 1],
                    branch_markers[branch_idx][step + 1],
                ))
            rng.shuffle(branches)
            branch_orders.append([f"branch{b[0]}:{b[1]}" for b in branches])
            for _, marker, branch_src_state, out_state, out_marker in branches:
                antecedent = f"{atom(marker, src_const)} & {atom(branch_src_state, src_const)}"
                if antecedent in seen_antecedents:
                    raise RuntimeError("duplicate hard_fsa_schema antecedent")
                seen_antecedents.add(antecedent)
                add_premise(
                    f"{antecedent} -> {atom(out_state, dst_const)}",
                    f"If {src_const} is {marker} and {src_const} is {branch_src_state}, then {dst_const} is {out_state}.",
                )
                add_premise(
                    f"{atom(out_state, dst_const)} -> {atom(out_marker, dst_const)}",
                    f"If {dst_const} is {out_state}, then {dst_const} is {out_marker}.",
                )

        total_premises = len(premises_fol)
        proof_entries: list[tuple[str, str]] = [(atom(branch_states[0][0], first_const), "R"), (atom(branch_markers[0][0], first_const), "R")]
        proof_nl_entries: list[str] = [f"{first_const} is {branch_states[0][0]}.", f"{first_const} is {branch_markers[0][0]}."]
        for step in range(depth):
            dst_const = constants[(step + 1) % len(constants)]
            proof_entries.append((atom(branch_states[0][step + 1], dst_const), "->E"))
            proof_nl_entries.append(f"{dst_const} is {branch_states[0][step + 1]}.")
            if step < depth - 1:
                proof_entries.append((atom(branch_markers[0][step + 1], dst_const), "->E"))
                proof_nl_entries.append(f"{dst_const} is {branch_markers[0][step + 1]}.")

        proof_fol = [ProofLine(total_premises + idx, formula, just).render() for idx, (formula, just) in enumerate(proof_entries, start=1)]
        proof_nl = [f"{total_premises + idx}. {text}" for idx, text in enumerate(proof_nl_entries, start=1)]
        final_const = constants[depth % len(constants)]
        answer = branch_states[0][-1]
        candidate_answers = [path[-1] for path in branch_states]
        wrong_candidates = [cand for cand in candidate_answers if cand != answer]
        rng.shuffle(wrong_candidates)
        gold_pos = index % branch_factor
        candidate_answers = wrong_candidates[:]
        candidate_answers.insert(gold_pos, answer)

        return LogicExample(
            constants=constants_lines,
            predicates=predicates_lines,
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol=f"Which state applies to {final_const}?",
            question_nl=f"Which state applies to {final_const}?",
            answer=answer,
            metadata={
                "depth": depth,
                "distractor_ratio": self.config.distractor_ratio,
                "num_distractors": branch_factor * depth,
                "difficulty": self.config.difficulty,
                "branching_factor": branch_factor,
                "shortcut_rate": self.config.shortcut_rate,
                "shortcut_enabled": True,
                "shortcut_types": ["marker_redundancy", "shared_transition_schema"],
                "split_intervention": "train_shortcut",
                "gold_answer": answer,
                "shortcut_branch_answer": branch_states[1][-1],
                "candidate_answers": candidate_answers,
                "gold_candidate_position": gold_pos,
                "path_states": branch_states[0],
                "path_markers": branch_markers[0],
                "path_families": [family_names[i] for i in branch_family_indices[0]],
                "branch_states": branch_states,
                "branch_markers": branch_markers,
                "branch_families": [[family_names[i] for i in fams] for fams in branch_family_indices],
                "path_constants": [constants[i % len(constants)] for i in range(depth + 1)],
                "branch_orders": branch_orders,
                "schema_predicted_family": family_names[gold_family_indices[-1]],
                "schema_predicted_answer": answer,
                "schema_prediction_correct": True,
                "marker_redundancy_correct": True,
                "citation_free_gold": True,
                "final_conclusion_kind": "state",
                "expected_proof_lines": 2 * depth + 1,
                "query_constant": final_const,
                "query_entity": final_const,
                "queried_family": "state",
            },
        )

    def generate(self, index: int) -> LogicExample:
        if self.config.is_hard_fsa_schema:
            return self._generate_hard_fsa_schema(index)
        if self.config.is_hard_fsa:
            return self._generate_hard_fsa(index)
        if self.config.is_hard_v5:
            return self._generate_hard_v5(index)
        rng = self._rng(index)
        families = self._select_families(rng)
        world, query_const = self._build_world(rng, families)
        chain_values = self._build_chain_values(rng, families, world, query_const)
        # Hard-v2 has adversarial premises that are aware of the gold chain.
        # The legacy distractor loop can accidentally add query-entity gold
        # facts, including the final answer, so keep it standard-only.
        num_distractors = 0 if self.config.is_hard else self._num_distractors()
        active_values = self._collect_active_values(
            families, chain_values, world, query_const
        )

        symbols = SymbolPool()
        symbols.constant_to_name = world.constants
        symbols.assign_predicates(active_values)

        constants_lines = [f"{c} = {name}" for c, name in world.constants.items()]
        predicates_lines = [f"{p}x: x is {v}" for v, p in symbols.value_to_pred.items()]

        premises_fol: List[str] = []
        premises_nl: List[str] = []
        proof_entries_fol: List[Tuple[str, str]] = []
        proof_entries_nl: List[str] = []

        premise_no = 1
        local_proof_no = 1

        # Base fact
        base_family = families[0]
        base_value = chain_values[base_family.name]
        base_pred = symbols.value_to_pred[base_value]
        base_atom = Atom(base_pred, query_const).render()

        premises_fol.append(f"{premise_no}. {base_atom}")
        premises_nl.append(f"{premise_no}. {world.constants[query_const]} is {base_value}.")
        base_premise_no = premise_no
        premise_no += 1

        # Proof line for base fact (use local proof references, remap to global line numbers later)
        proof_entries_fol.append((base_atom, f"R,P{base_premise_no}"))
        proof_entries_nl.append(f"{local_proof_no}. {world.constants[query_const]} is {base_value}.")
        formula_to_proof_ref = {base_atom: local_proof_no}
        current_formula = base_atom
        current_proof_ref = local_proof_no
        local_proof_no += 1

        # Structured rules
        rule_instances = self._build_rule_instances(families, chain_values, symbols.value_to_pred)

        for step, (kind, rule_inst) in enumerate(rule_instances, start=1):
            if kind == "implication":
                src_pred = rule_inst.source_preds[0]
                rule_formula = f"{src_pred}{query_const} -> {rule_inst.target_pred}{query_const}"
                premises_fol.append(f"{premise_no}. {rule_formula}")
                premises_nl.append(f"{premise_no}. {rule_inst.nl()}")
                rule_premise_no = premise_no
                premise_no += 1

                conclusion = Atom(rule_inst.target_pred, query_const).render()
                proof_entries_fol.append(
                    (conclusion, f"->E,P{rule_premise_no},L{current_proof_ref}")
                )
                proof_entries_nl.append(
                    f"{local_proof_no}. Since {world.constants[query_const]} is {rule_inst.source_texts[0]}, "
                    f"{world.constants[query_const]} is {rule_inst.target_text}."
                )
                current_formula = conclusion
                current_proof_ref = local_proof_no
                formula_to_proof_ref.setdefault(conclusion, local_proof_no)
                local_proof_no += 1

            else:
                # Hard mode should not re-prove facts that are already in the
                # gold proof. Reusing earlier lines keeps the target proof near
                # shortest-path while distractors carry the difficulty.
                side_value = rule_inst.source_texts[1]
                side_pred = rule_inst.source_preds[1]
                side_atom = Atom(side_pred, query_const).render()
                if self.config.is_hard and side_atom in formula_to_proof_ref:
                    side_proof_ref = formula_to_proof_ref[side_atom]
                else:
                    side_chain_depth = self.config.effective_side_chain_depth if self.config.is_hard else 0
                    chain_preds = {symbols.value_to_pred[chain_values[fam.name]] for fam in families}
                    reserved_preds = chain_preds | {rule_inst.target_pred}
                    side_intermediates = [
                        p
                        for p in symbols.value_to_pred.values()
                        if p not in reserved_preds
                    ][:side_chain_depth]
                    if side_chain_depth > 0 and side_intermediates:
                        support_preds = side_intermediates + [side_pred]
                    else:
                        support_preds = []

                    if support_preds:
                        side_base_pred = support_preds[0]
                        side_base_value = self._value_for_pred(symbols.value_to_pred, side_base_pred)
                        side_base_atom = Atom(side_base_pred, query_const).render()

                        premises_fol.append(f"{premise_no}. {side_base_atom}")
                        premises_nl.append(f"{premise_no}. {world.constants[query_const]} is {side_base_value}.")
                        side_fact_premise_no = premise_no
                        premise_no += 1

                        proof_entries_fol.append((side_base_atom, f"R,P{side_fact_premise_no}"))
                        proof_entries_nl.append(f"{local_proof_no}. {world.constants[query_const]} is {side_base_value}.")
                        side_proof_ref = local_proof_no
                        formula_to_proof_ref.setdefault(side_base_atom, local_proof_no)
                        local_proof_no += 1

                        prev_pred = side_base_pred
                        prev_text = side_base_value
                        for next_pred in support_preds[1:]:
                            next_text = self._value_for_pred(symbols.value_to_pred, next_pred)
                            rule_formula = f"{prev_pred}{query_const} -> {next_pred}{query_const}"
                            premises_fol.append(f"{premise_no}. {rule_formula}")
                            premises_nl.append(f"{premise_no}. All things that are {prev_text} are {next_text}.")
                            side_rule_premise_no = premise_no
                            premise_no += 1

                            next_atom = Atom(next_pred, query_const).render()
                            proof_entries_fol.append((next_atom, f"->E,P{side_rule_premise_no},L{side_proof_ref}"))
                            proof_entries_nl.append(
                                f"{local_proof_no}. Since {world.constants[query_const]} is {prev_text}, "
                                f"{world.constants[query_const]} is {next_text}."
                            )
                            side_proof_ref = local_proof_no
                            formula_to_proof_ref.setdefault(next_atom, local_proof_no)
                            local_proof_no += 1
                            prev_pred = next_pred
                            prev_text = next_text
                    else:
                        premises_fol.append(f"{premise_no}. {side_atom}")
                        premises_nl.append(f"{premise_no}. {world.constants[query_const]} is {side_value}.")
                        side_fact_premise_no = premise_no
                        premise_no += 1

                        proof_entries_fol.append((side_atom, f"R,P{side_fact_premise_no}"))
                        proof_entries_nl.append(f"{local_proof_no}. {world.constants[query_const]} is {side_value}.")
                        side_proof_ref = local_proof_no
                        formula_to_proof_ref.setdefault(side_atom, local_proof_no)
                        local_proof_no += 1

                src_left, src_right = rule_inst.source_preds
                rule_formula = (
                    f"{src_left}{query_const} & {src_right}{query_const} -> {rule_inst.target_pred}{query_const}"
                )
                premises_fol.append(f"{premise_no}. {rule_formula}")
                premises_nl.append(f"{premise_no}. {rule_inst.nl()}")
                rule_premise_no = premise_no
                premise_no += 1

                conjunction_formula = f"{current_formula} & {side_atom}"
                proof_entries_fol.append(
                    (conjunction_formula, f"CI,L{current_proof_ref},L{side_proof_ref}")
                )
                proof_entries_nl.append(
                    f"{local_proof_no}. {world.constants[query_const]} is both {rule_inst.source_texts[0]} and {rule_inst.source_texts[1]}."
                )
                conjunction_proof_ref = local_proof_no
                formula_to_proof_ref.setdefault(conjunction_formula, local_proof_no)
                local_proof_no += 1

                conclusion = Atom(rule_inst.target_pred, query_const).render()
                proof_entries_fol.append(
                    (
                        conclusion,
                        f"->E,P{rule_premise_no},L{conjunction_proof_ref}",
                    )
                )
                proof_entries_nl.append(
                    f"{local_proof_no}. Since {world.constants[query_const]} is {rule_inst.source_texts[0]} "
                    f"and {rule_inst.source_texts[1]}, {world.constants[query_const]} is {rule_inst.target_text}."
                )
                current_formula = conclusion
                current_proof_ref = local_proof_no
                formula_to_proof_ref.setdefault(conclusion, local_proof_no)
                local_proof_no += 1

        hard_counts: dict[str, Any] = {}
        if self.config.is_hard:
            premise_no, hard_counts = self._append_hard_premises(
                rng=rng,
                families=families,
                world=world,
                query_const=query_const,
                chain_values=chain_values,
                value_to_pred=symbols.value_to_pred,
                premises_fol=premises_fol,
                premises_nl=premises_nl,
                premise_no=premise_no,
            )

        # Distractors: clearly structured, same rule templates
        existing_premise_formulas = {
            line.split(". ", 1)[1].strip() if ". " in line else line.strip()
            for line in premises_fol
        }
        distractor_kinds = ["fact", "implication"]
        if len(families) >= 3:
            distractor_kinds.append("conjunction")
        added_distractors = 0
        attempts = 0
        max_attempts = max(16, num_distractors * 16)
        while added_distractors < num_distractors and attempts < max_attempts:
            attempts += 1
            distractor_kind = rng.choice(distractor_kinds)
            if distractor_kind == "fact":
                c = query_const
                fam = rng.choice(families)
                val = chain_values[fam.name]
                pred = symbols.value_to_pred[val]
                formula = Atom(pred, c).render()
                if formula in existing_premise_formulas:
                    continue
                premises_fol.append(f"{premise_no}. {formula}")
                premises_nl.append(f"{premise_no}. {world.constants[c]} is {val}.")
                existing_premise_formulas.add(formula)
                premise_no += 1
                added_distractors += 1

            elif distractor_kind == "implication":
                fam1, fam2 = rng.sample(families, 2)
                v1 = chain_values[fam1.name]
                v2 = chain_values[fam2.name]
                if v1 not in symbols.value_to_pred or v2 not in symbols.value_to_pred:
                    continue
                inst = RuleInstance(
                    template=RULES["implication"],
                    source_preds=(symbols.value_to_pred[v1],),
                    target_pred=symbols.value_to_pred[v2],
                    source_texts=(v1,),
                    target_text=v2,
                )
                formula = f"{inst.source_preds[0]}{query_const} -> {inst.target_pred}{query_const}"
                if formula in existing_premise_formulas:
                    continue
                premises_fol.append(f"{premise_no}. {formula}")
                premises_nl.append(f"{premise_no}. {inst.nl()}")
                existing_premise_formulas.add(formula)
                premise_no += 1
                added_distractors += 1

            else:
                fam1, fam2, fam3 = rng.sample(families, 3)
                v1 = chain_values[fam1.name]
                v2 = chain_values[fam2.name]
                v3 = chain_values[fam3.name]
                if any(v not in symbols.value_to_pred for v in (v1, v2, v3)):
                    continue
                inst = RuleInstance(
                    template=RULES["conjunction"],
                    source_preds=(symbols.value_to_pred[v1], symbols.value_to_pred[v2]),
                    target_pred=symbols.value_to_pred[v3],
                    source_texts=(v1, v2),
                    target_text=v3,
                )
                formula = (
                    f"{inst.source_preds[0]}{query_const} & {inst.source_preds[1]}{query_const} -> "
                    f"{inst.target_pred}{query_const}"
                )
                if formula in existing_premise_formulas:
                    continue
                premises_fol.append(f"{premise_no}. {formula}")
                premises_nl.append(f"{premise_no}. {inst.nl()}")
                existing_premise_formulas.add(formula)
                premise_no += 1
                added_distractors += 1

        total_premises = len(premises_fol)
        proof_fol = []
        for idx, (formula, justification_template) in enumerate(proof_entries_fol, start=1):
            line_number = total_premises + idx
            justification = self._render_justification(justification_template, total_premises)
            proof_fol.append(ProofLine(line_number, formula, justification).render())

        proof_nl = [
            f"{total_premises + idx}. {line.split('. ', 1)[1]}"
            for idx, line in enumerate(proof_entries_nl, start=1)
        ]

        query_family = families[-1]
        question_fol = f"What value of {query_family.name} does {query_const} have?"
        question_nl = query_family.question_template.format(entity=world.constants[query_const])
        answer = chain_values[query_family.name]
        nl_premises_shuffled = self.config.is_hard_v3
        if nl_premises_shuffled:
            premises_nl = self._shuffle_natural_theory(rng, premises_nl)

        return LogicExample(
            constants=constants_lines,
            predicates=predicates_lines,
            premises_fol=premises_fol,
            premises_nl=premises_nl,
            proof_fol=proof_fol,
            proof_nl=proof_nl,
            question_fol=question_fol,
            question_nl=question_nl,
            answer=answer,
            metadata={
                "depth": self.config.depth,
                "distractor_ratio": self.config.distractor_ratio,
                "num_distractors": added_distractors,
                "difficulty": self.config.difficulty,
                "branching_factor": self.config.effective_branching_factor,
                "decoy_chains": self.config.effective_decoy_chains,
                "near_miss_ratio": self.config.effective_near_miss_ratio,
                "side_chain_depth": self.config.effective_side_chain_depth,
                "entity_decoy_ratio": self.config.effective_entity_decoy_ratio,
                "answer_decoy_ratio": self.config.effective_answer_decoy_ratio,
                "hard_counts": hard_counts,
                "nl_premises_shuffled": nl_premises_shuffled,
                "query_constant": query_const,
                "query_entity": world.constants[query_const],
                "queried_family": query_family.name,
            },
        )


# ============================================================
# Hugging Face friendly interface
# ============================================================

def example_stream(config: DatasetConfig, start_index: int = 0) -> Iterator[Dict[str, Any]]:
    gen = LogicDatasetGenerator(config)
    idx = start_index
    while True:
        yield gen.generate(idx).to_dict()
        idx += 1

def finite_example_stream(config: DatasetConfig, n: int, start_index: int = 0) -> Iterator[Dict[str, Any]]:
    gen = LogicDatasetGenerator(config)
    for idx in range(start_index, start_index + n):
        yield gen.generate(idx).to_dict()

def build_hf_iterable_dataset(config: DatasetConfig):
    try:
        from datasets import IterableDataset
    except ImportError as e:
        raise ImportError("Install `datasets` to use build_hf_iterable_dataset.") from e
    return IterableDataset.from_generator(example_stream, gen_kwargs={"config": config})

def build_hf_validation_dataset(config: DatasetConfig, n: int):
    try:
        from datasets import IterableDataset
    except ImportError as e:
        raise ImportError("Install `datasets` to use build_hf_validation_dataset.") from e
    return IterableDataset.from_generator(finite_example_stream, gen_kwargs={"config": config, "n": n})


# ============================================================
# Materialized dataset access
# ============================================================

@dataclass(frozen=True)
class MaterializedSyntheticDataset:
    train_up_to_3_subset: str = "train_up_to_3_1k"
    train_up_to_5_subset: str = "train_up_to_5_1m"
    train_up_to_10_subset: str = "train_up_to_10_1m"
    train_up_to_15_subset: str = "train_up_to_15_120k"

    def val_subset_name(self, step: int) -> str:
        return f"val_step_{step:02d}_1k"

    def train_subset_for_max_step(self, max_step: int) -> str:
        if max_step <= 3:
            return self.train_up_to_3_subset
        if max_step <= 5:
            return self.train_up_to_5_subset
        if max_step <= 10:
            return self.train_up_to_10_subset
        return self.train_up_to_15_subset

    def materialized_parquet_path(self, local_root: str | Path, subset: str) -> Path:
        return Path(local_root).expanduser().resolve() / subset / "train.parquet"

    def load_rows(
        self,
        *,
        subset: str,
        dataset_id: str | None = None,
        local_root: str | None = None,
        split: str = "train",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Install `datasets` to load materialized data.") from e

        split_spec = split
        if limit is not None:
            split_spec = f"{split}[:{int(limit)}]"

        if dataset_id:
            ds = load_dataset(dataset_id, subset, split=split_spec)
        else:
            if not local_root:
                raise ValueError("Either dataset_id or local_root must be provided for materialized data.")
            parquet_file = self.materialized_parquet_path(local_root, subset)
            if not parquet_file.exists():
                raise FileNotFoundError(f"Materialized parquet not found: {parquet_file}")
            ds = load_dataset("parquet", data_files={split: str(parquet_file)}, split=split_spec)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        return ds.to_list()


@dataclass(frozen=True)
class MaterializedDatasetSpec:
    subset: str
    min_depth: int
    max_depth: int
    rows: int
    seed: int
    shortcut_rate: float = 0.0


class MaterializedDatasetBuilder:
    def __init__(self, dataset: MaterializedSyntheticDataset | None = None) -> None:
        self.dataset = dataset or MaterializedSyntheticDataset()

    def train_specs(
        self,
        *,
        train_up_to_3_rows: int,
        train_up_to_5_rows: int,
        train_up_to_10_rows: int,
        train_up_to_15_rows: int,
        seed: int,
        train_shortcut_rate: float = 0.0,
    ) -> list[MaterializedDatasetSpec]:
        specs = [
            MaterializedDatasetSpec(
                subset=self.dataset.train_up_to_3_subset,
                min_depth=1,
                max_depth=3,
                rows=int(train_up_to_3_rows),
                seed=int(seed) - 100_000,
                shortcut_rate=float(train_shortcut_rate),
            ),
            MaterializedDatasetSpec(
                subset=self.dataset.train_up_to_5_subset,
                min_depth=1,
                max_depth=5,
                rows=int(train_up_to_5_rows),
                seed=int(seed),
                shortcut_rate=float(train_shortcut_rate),
            ),
            MaterializedDatasetSpec(
                subset=self.dataset.train_up_to_10_subset,
                min_depth=1,
                max_depth=10,
                rows=int(train_up_to_10_rows),
                seed=int(seed) + 100_000,
                shortcut_rate=float(train_shortcut_rate),
            ),
            MaterializedDatasetSpec(
                subset=self.dataset.train_up_to_15_subset,
                min_depth=1,
                max_depth=15,
                rows=int(train_up_to_15_rows),
                seed=int(seed) + 200_000,
                shortcut_rate=float(train_shortcut_rate),
            ),
        ]
        return [spec for spec in specs if spec.rows > 0]

    def val_specs(
        self,
        *,
        val_rows_per_step: int,
        seed: int,
        val_shortcut_rate: float = 0.0,
        val_max_step: int = 20,
    ) -> list[MaterializedDatasetSpec]:
        specs: list[MaterializedDatasetSpec] = []
        for step in range(1, int(val_max_step) + 1):
            specs.append(
                MaterializedDatasetSpec(
                    subset=self.dataset.val_subset_name(step),
                    min_depth=step,
                    max_depth=step,
                    rows=int(val_rows_per_step),
                    seed=int(seed) + 1_000_000 + step * 10_000,
                    shortcut_rate=float(val_shortcut_rate),
                )
            )
        return [spec for spec in specs if spec.rows > 0]

    def build(
        self,
        *,
        output_root: str | Path,
        train_up_to_3_rows: int = 0,
        train_up_to_5_rows: int = 1_000_000,
        train_up_to_10_rows: int = 1_000_000,
        train_up_to_15_rows: int = 0,
        val_rows_per_step: int = 1_000,
        seed: int = 3407,
        distractor_ratio: float = 0.5,
        difficulty: str = "standard",
        train_shortcut_rate: float = 0.0,
        val_shortcut_rate: float = 0.0,
        val_max_step: int = 20,
        branching_factor: int | None = None,
        decoy_chains: int | None = None,
        near_miss_ratio: float | None = None,
        side_chain_depth: int | None = None,
        entity_decoy_ratio: float | None = None,
        answer_decoy_ratio: float | None = None,
        chunk_size: int = 10_000,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_root = Path(output_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        specs = self.train_specs(
            train_up_to_3_rows=int(train_up_to_3_rows),
            train_up_to_5_rows=int(train_up_to_5_rows),
            train_up_to_10_rows=int(train_up_to_10_rows),
            train_up_to_15_rows=int(train_up_to_15_rows),
            seed=int(seed),
            train_shortcut_rate=float(train_shortcut_rate),
        ) + self.val_specs(
            val_rows_per_step=int(val_rows_per_step),
            seed=int(seed),
            val_shortcut_rate=float(val_shortcut_rate),
            val_max_step=int(val_max_step),
        )

        for spec in specs:
            out_file = self.dataset.materialized_parquet_path(out_root, spec.subset)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            writer: pq.ParquetWriter | None = None
            chunk: list[dict[str, Any]] = []
            for row in self._records_for_spec(
                spec=spec,
                distractor_ratio=float(distractor_ratio),
                difficulty=str(difficulty),
                branching_factor=branching_factor,
                decoy_chains=decoy_chains,
                near_miss_ratio=near_miss_ratio,
                side_chain_depth=side_chain_depth,
                entity_decoy_ratio=entity_decoy_ratio,
                answer_decoy_ratio=answer_decoy_ratio,
                shortcut_rate=float(spec.shortcut_rate),
            ):
                chunk.append(row)
                if len(chunk) >= int(chunk_size):
                    table = pa.Table.from_pylist(chunk)
                    if writer is None:
                        writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
                    writer.write_table(table)
                    chunk = []
            if chunk:
                table = pa.Table.from_pylist(chunk)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
                writer.write_table(table)
            if writer is not None:
                writer.close()

        manifest = {
            "train_subsets": [spec.subset for spec in specs if spec.subset.startswith("train_")],
            "val_subsets": [self.dataset.val_subset_name(i) for i in range(1, int(val_max_step) + 1)],
            "difficulty": difficulty,
            "distractor_ratio": distractor_ratio,
            "train_shortcut_rate": train_shortcut_rate,
            "val_shortcut_rate": val_shortcut_rate,
            "branching_factor": branching_factor,
            "decoy_chains": decoy_chains,
            "near_miss_ratio": near_miss_ratio,
            "side_chain_depth": side_chain_depth,
            "entity_decoy_ratio": entity_decoy_ratio,
            "answer_decoy_ratio": answer_decoy_ratio,
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def push_to_hub(
        self,
        *,
        output_root: str | Path,
        repo_id: str,
        private: bool = False,
    ) -> None:
        import time

        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError("Install `datasets` to push materialized data to Hub.") from e

        root = Path(output_root).expanduser().resolve()
        configured_subsets = [
            self.dataset.train_up_to_3_subset,
            self.dataset.train_up_to_5_subset,
            self.dataset.train_up_to_10_subset,
            self.dataset.train_up_to_15_subset,
        ] + [
            self.dataset.val_subset_name(i) for i in range(1, 21)
        ]
        available_subsets = sorted(p.parent.name for p in root.glob("*/train.parquet"))
        subsets = list(dict.fromkeys(configured_subsets + available_subsets))
        for subset in subsets:
            parquet_file = self.dataset.materialized_parquet_path(root, subset)
            if not parquet_file.exists():
                continue
            ds = Dataset.from_parquet(str(parquet_file))
            last_error: Exception | None = None
            for attempt in range(1, 6):
                try:
                    ds.push_to_hub(repo_id=repo_id, config_name=subset, split="train", private=bool(private))
                    last_error = None
                    break
                except Exception as exc:  # HF occasionally returns transient 5xx on repo_info/upload.
                    last_error = exc
                    if attempt == 5:
                        break
                    sleep_s = min(300, 20 * attempt)
                    print(f"[push_to_hub] subset={subset} attempt={attempt} failed: {exc}; retrying in {sleep_s}s", flush=True)
                    time.sleep(sleep_s)
            if last_error is not None:
                raise last_error

    @staticmethod
    def _core_record(gen: LogicDatasetGenerator, index: int) -> dict[str, Any]:
        ex = gen.generate(index)
        row = ex.to_dict()
        row["depth"] = int(ex.metadata.get("depth", gen.config.depth))
        row["record_index"] = int(index)
        return row

    def _records_for_spec(
        self,
        *,
        spec: MaterializedDatasetSpec,
        distractor_ratio: float,
        difficulty: str = "standard",
        branching_factor: int | None = None,
        decoy_chains: int | None = None,
        near_miss_ratio: float | None = None,
        side_chain_depth: int | None = None,
        entity_decoy_ratio: float | None = None,
        answer_decoy_ratio: float | None = None,
        shortcut_rate: float = 0.0,
    ) -> Iterator[dict[str, Any]]:
        depths = list(range(spec.min_depth, spec.max_depth + 1))
        gens = {
            depth: LogicDatasetGenerator(
                DatasetConfig(
                    depth=depth,
                    distractor_ratio=distractor_ratio,
                    difficulty=difficulty,
                    branching_factor=branching_factor,
                    decoy_chains=decoy_chains,
                    near_miss_ratio=near_miss_ratio,
                    side_chain_depth=side_chain_depth,
                    entity_decoy_ratio=entity_decoy_ratio,
                    answer_decoy_ratio=answer_decoy_ratio,
                    shortcut_rate=float(shortcut_rate),
                    seed=spec.seed + depth,
                )
            )
            for depth in depths
        }
        counters = {depth: 0 for depth in depths}
        for i in range(spec.rows):
            depth = depths[i % len(depths)]
            index = counters[depth]
            counters[depth] += 1
            yield self._core_record(gens[depth], index)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    cfg = DatasetConfig(depth=4, distractor_ratio=0.5, min_entities=4, max_entities=6, seed=7)
    gen = LogicDatasetGenerator(cfg)
    ex = gen.generate(0)
    print(json.dumps(ex.to_dict(), indent=2, ensure_ascii=False))

    from logic_engine import LogicEngine
    engine = LogicEngine()
    valid = engine.validate_proof(
        premises="\n".join(ex.premises_fol),
        conclusion=ex.proof_fol[-1].split(". ", 1)[1].split(" ; ", 1)[0],
        proof="\n".join(ex.proof_fol)
    )

    print(f"Proof valid? {valid}")
