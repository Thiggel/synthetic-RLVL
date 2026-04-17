
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
    min_entities: int = 4
    max_entities: int = 8
    num_attribute_values: int = 8
    seed: int = 0

    def __post_init__(self):
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
        out, seen = [], set()
        for v in values:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

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

    def generate(self, index: int) -> LogicExample:
        rng = self._rng(index)
        families = self._select_families(rng)
        world, query_const = self._build_world(rng, families)
        chain_values = self._build_chain_values(rng, families, world, query_const)
        num_distractors = self._num_distractors()
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
                local_proof_no += 1

            else:
                # Add side fact
                side_value = rule_inst.source_texts[1]
                side_pred = rule_inst.source_preds[1]
                side_atom = Atom(side_pred, query_const).render()

                premises_fol.append(f"{premise_no}. {side_atom}")
                premises_nl.append(f"{premise_no}. {world.constants[query_const]} is {side_value}.")
                side_fact_premise_no = premise_no
                premise_no += 1

                proof_entries_fol.append(
                    (side_atom, f"R,P{side_fact_premise_no}")
                )
                proof_entries_nl.append(f"{local_proof_no}. {world.constants[query_const]} is {side_value}.")
                side_proof_ref = local_proof_no
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
                local_proof_no += 1

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
    train_up_to_5_subset: str = "train_up_to_5_1m"
    train_up_to_10_subset: str = "train_up_to_10_1m"

    def val_subset_name(self, step: int) -> str:
        return f"val_step_{step:02d}_1k"

    def train_subset_for_max_step(self, max_step: int) -> str:
        if max_step <= 5:
            return self.train_up_to_5_subset
        return self.train_up_to_10_subset

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


class MaterializedDatasetBuilder:
    def __init__(self, dataset: MaterializedSyntheticDataset | None = None) -> None:
        self.dataset = dataset or MaterializedSyntheticDataset()

    def train_specs(
        self,
        *,
        train_up_to_5_rows: int,
        train_up_to_10_rows: int,
        seed: int,
    ) -> list[MaterializedDatasetSpec]:
        return [
            MaterializedDatasetSpec(
                subset=self.dataset.train_up_to_5_subset,
                min_depth=1,
                max_depth=5,
                rows=int(train_up_to_5_rows),
                seed=int(seed),
            ),
            MaterializedDatasetSpec(
                subset=self.dataset.train_up_to_10_subset,
                min_depth=1,
                max_depth=10,
                rows=int(train_up_to_10_rows),
                seed=int(seed) + 100_000,
            ),
        ]

    def val_specs(self, *, val_rows_per_step: int, seed: int) -> list[MaterializedDatasetSpec]:
        specs: list[MaterializedDatasetSpec] = []
        for step in range(1, 21):
            specs.append(
                MaterializedDatasetSpec(
                    subset=self.dataset.val_subset_name(step),
                    min_depth=step,
                    max_depth=step,
                    rows=int(val_rows_per_step),
                    seed=int(seed) + 1_000_000 + step * 10_000,
                )
            )
        return specs

    def build(
        self,
        *,
        output_root: str | Path,
        train_up_to_5_rows: int = 1_000_000,
        train_up_to_10_rows: int = 1_000_000,
        val_rows_per_step: int = 1_000,
        seed: int = 3407,
        distractor_ratio: float = 0.5,
        chunk_size: int = 10_000,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_root = Path(output_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        specs = self.train_specs(
            train_up_to_5_rows=int(train_up_to_5_rows),
            train_up_to_10_rows=int(train_up_to_10_rows),
            seed=int(seed),
        ) + self.val_specs(val_rows_per_step=int(val_rows_per_step), seed=int(seed))

        for spec in specs:
            out_file = self.dataset.materialized_parquet_path(out_root, spec.subset)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            writer: pq.ParquetWriter | None = None
            chunk: list[dict[str, Any]] = []
            for row in self._records_for_spec(spec=spec, distractor_ratio=float(distractor_ratio)):
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
            "train_subsets": [self.dataset.train_up_to_5_subset, self.dataset.train_up_to_10_subset],
            "val_subsets": [self.dataset.val_subset_name(i) for i in range(1, 21)],
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def push_to_hub(
        self,
        *,
        output_root: str | Path,
        repo_id: str,
        private: bool = False,
    ) -> None:
        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError("Install `datasets` to push materialized data to Hub.") from e

        root = Path(output_root).expanduser().resolve()
        subsets = [self.dataset.train_up_to_5_subset, self.dataset.train_up_to_10_subset] + [
            self.dataset.val_subset_name(i) for i in range(1, 21)
        ]
        for subset in subsets:
            parquet_file = self.dataset.materialized_parquet_path(root, subset)
            ds = Dataset.from_parquet(str(parquet_file))
            ds.push_to_hub(repo_id=repo_id, config_name=subset, split="train", private=bool(private))

    @staticmethod
    def _core_record(gen: LogicDatasetGenerator, index: int) -> dict[str, Any]:
        ex = gen.generate(index)
        row = ex.to_dict()
        row["depth"] = int(ex.metadata.get("depth", gen.config.depth))
        row["record_index"] = int(index)
        return row

    def _records_for_spec(self, *, spec: MaterializedDatasetSpec, distractor_ratio: float) -> Iterator[dict[str, Any]]:
        depths = list(range(spec.min_depth, spec.max_depth + 1))
        gens = {
            depth: LogicDatasetGenerator(
                DatasetConfig(
                    depth=depth,
                    distractor_ratio=distractor_ratio,
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
