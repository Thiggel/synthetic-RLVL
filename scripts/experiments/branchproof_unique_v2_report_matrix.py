#!/usr/bin/env python
"""Canonical corrected BranchProof-v2 report rerun matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


SEEDS = (3407, 3408, 3409)
TRAIN_MAXES = (5, 10, 15, 20, 25)
BASE_MODEL = "allenai/Olmo-3-1025-7B"


@dataclass(frozen=True)
class TrainRow:
    family: str
    run_name: str
    train_template: str
    eval_templates: tuple[str, ...]
    seed: int
    train_max: int
    model_name: str = BASE_MODEL
    max_steps: int = 10_000
    per_device_batch_size: int = 1
    grad_accum: int = 1
    balanced_modality_batches: bool = False
    data_kind: str = "base"
    shortcut_rate: float = 0.0
    shortcut_kind: str = "schema"
    max_length: int = 8192

    @property
    def train_subset(self) -> str:
        return f"train_fixedtarget_up_to_{self.train_max}_50k"


@dataclass(frozen=True)
class EvalRow:
    train_index: int
    train: TrainRow
    eval_template: str

    @property
    def output_tag(self) -> str:
        if self.eval_template == self.train.train_template:
            return self.train.run_name
        return f"{self.train.run_name}_{self.eval_template}"


def _basic_row(
    *,
    family: str,
    stem: str,
    template: str,
    seed: int,
    train_max: int = 25,
    **kwargs,
) -> TrainRow:
    return TrainRow(
        family=family,
        run_name=f"sft_branchproof_unique_v2_{stem}_seed{seed}",
        train_template=template,
        eval_templates=(template,),
        seed=seed,
        train_max=train_max,
        **kwargs,
    )


def surface_rows() -> list[TrainRow]:
    rows: list[TrainRow] = []
    templates = (
        "logic_symbol_padded",
        "logic_wordified",
        "terse_nl",
        "rule_annotated_nl",
        "pseudocode",
        "shuffled_logic",
        "invalid_logic",
        "shuffled_nl",
    )
    for template in templates:
        for seed in SEEDS:
            rows.append(
                _basic_row(
                    family="surface",
                    stem=f"surface_{template}_train1to25_10k",
                    template=template,
                    seed=seed,
                )
            )
    # The matched formal row is exactly the corrected main train-1-to-25
    # baseline, so only the shorter NL exposure requires a new run.
    for seed in SEEDS:
        rows.append(
            _basic_row(
                family="surface",
                stem="same_target_tokens_nl_exact_train1to25_7140steps",
                template="nl_exact",
                seed=seed,
                max_steps=7140,
            )
        )
    return rows


def shortcut_rows() -> list[TrainRow]:
    rows: list[TrainRow] = []
    specifications = [
        ("schema", "0p3", 0.3),
        ("schema", "0p5", 0.5),
        ("schema", "0p8", 0.8),
        ("position", "0p5", 0.5),
        ("position", "0p8", 0.8),
        ("initial_marker", "0p5", 0.5),
        ("initial_marker", "0p8", 0.8),
    ]
    for kind, rate_tag, rate in specifications:
        data_kind = f"shortcut_{kind}_{rate_tag}"
        for template in ("logic", "nl_exact"):
            for seed in SEEDS:
                rows.append(
                    _basic_row(
                        family="shortcut",
                        stem=f"shortcut_{kind}_{rate_tag}_{template}_train1to25_10k",
                        template=template,
                        seed=seed,
                        data_kind=data_kind,
                        shortcut_rate=rate,
                        shortcut_kind=kind,
                    )
                )
    return rows


def hybrid_rows() -> list[TrainRow]:
    rows: list[TrainRow] = []
    for template in ("think_formal", "formal_think"):
        for train_max in TRAIN_MAXES:
            for seed in SEEDS:
                rows.append(
                    _basic_row(
                        family="hybrid",
                        stem=f"hybrid_{template}_train1to{train_max}_10k",
                        template=template,
                        seed=seed,
                        train_max=train_max,
                        max_length=16384,
                    )
                )
    return rows


def conditioned_rows(*, family: str, final_steps: int) -> list[TrainRow]:
    rows: list[TrainRow] = []
    for train_max in TRAIN_MAXES:
        for seed in SEEDS:
            rows.append(
                TrainRow(
                    family=family,
                    run_name=(
                        f"sft_branchproof_unique_v2_{family}_train1to{train_max}_"
                        f"{final_steps // 1000}k_seed{seed}"
                    ),
                    train_template="conditioned_dual",
                    eval_templates=("conditioned_logic", "conditioned_nl"),
                    seed=seed,
                    train_max=train_max,
                    max_steps=final_steps,
                )
            )
    return rows


def architecture_rows() -> list[TrainRow]:
    rows: list[TrainRow] = []
    models = (
        ("qwen2p5_1p5b", "Qwen/Qwen2.5-1.5B"),
        ("qwen2p5_7b", "Qwen/Qwen2.5-7B"),
        ("gemma3_4b_pt", "google/gemma-3-4b-pt"),
    )
    for model_tag, model_name in models:
        for template in ("logic", "nl_exact"):
            for train_max in (10, 20, 25):
                for seed in SEEDS:
                    rows.append(
                        _basic_row(
                            family="architecture",
                            stem=f"arch_{model_tag}_{template}_train1to{train_max}_10k",
                            template=template,
                            seed=seed,
                            train_max=train_max,
                            model_name=model_name,
                        )
                    )
    return rows


def batch_rows() -> list[TrainRow]:
    rows: list[TrainRow] = []
    for condition in ("logic", "nl_exact", "conditioned_dual"):
        for batch_size in (2, 4, 8, 16):
            for seed in SEEDS:
                per_device = min(batch_size, 8)
                grad_accum = batch_size // per_device
                eval_templates = (
                    ("conditioned_logic", "conditioned_nl")
                    if condition == "conditioned_dual"
                    else (condition,)
                )
                rows.append(
                    TrainRow(
                        family="batch",
                        run_name=(
                            f"sft_branchproof_unique_v2_batch_bsz{batch_size}_{condition}_"
                            f"train1to20_10k_seed{seed}"
                        ),
                        train_template=condition,
                        eval_templates=eval_templates,
                        seed=seed,
                        train_max=20,
                        per_device_batch_size=per_device,
                        grad_accum=grad_accum,
                        balanced_modality_batches=condition == "conditioned_dual",
                    )
                )
    return rows


def large_rows() -> list[TrainRow]:
    rows: list[TrainRow] = []
    models = (
        ("olmo3_1125_32b", "allenai/Olmo-3-1125-32B"),
        ("qwen3_32b", "Qwen/Qwen3-32B"),
    )
    for model_tag, model_name in models:
        for template in ("logic", "nl_exact"):
            for seed in SEEDS:
                rows.append(
                    _basic_row(
                        family="large",
                        stem=f"arch_{model_tag}_{template}_train1to25_10k",
                        template=template,
                        seed=seed,
                        model_name=model_name,
                    )
                )
    for seed in SEEDS:
        rows.append(
            TrainRow(
                family="large",
                run_name=f"sft_branchproof_unique_v2_arch_olmo3_1125_32b_conditioned_dual_train1to25_10k_seed{seed}",
                train_template="conditioned_dual",
                eval_templates=("conditioned_logic", "conditioned_nl"),
                seed=seed,
                train_max=25,
                model_name="allenai/Olmo-3-1125-32B",
                grad_accum=2,
                balanced_modality_batches=True,
            )
        )
    return rows


GROUP_BUILDERS = {
    "surface": surface_rows,
    "shortcut": shortcut_rows,
    "hybrid": hybrid_rows,
    "conditioned10k": lambda: conditioned_rows(family="conditioned10k", final_steps=10_000),
    "conditioned50k": lambda: conditioned_rows(family="conditioned50k", final_steps=50_000),
    "architecture": architecture_rows,
    "batch": batch_rows,
    "large": large_rows,
}


def train_rows(group: str) -> list[TrainRow]:
    return GROUP_BUILDERS[group]()


def eval_rows(group: str) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for train_index, train in enumerate(train_rows(group)):
        rows.extend(EvalRow(train_index, train, template) for template in train.eval_templates)
    return rows


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _train_tsv(row: TrainRow) -> str:
    fields = (
        row.family,
        row.run_name,
        row.train_template,
        ":".join(row.eval_templates),
        str(row.seed),
        str(row.train_max),
        row.train_subset,
        row.model_name,
        str(row.max_steps),
        str(row.per_device_batch_size),
        str(row.grad_accum),
        _bool(row.balanced_modality_batches),
        row.data_kind,
        str(row.shortcut_rate),
        row.shortcut_kind,
        str(row.max_length),
    )
    return "\t".join(fields)


def _eval_tsv(row: EvalRow) -> str:
    return "\t".join(
        (
            str(row.train_index),
            _train_tsv(row.train),
            row.eval_template,
            row.output_tag,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(GROUP_BUILDERS))
    parser.add_argument("--kind", required=True, choices=("train", "eval"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()

    rows = train_rows(args.group) if args.kind == "train" else eval_rows(args.group)
    if args.count:
        print(len(rows))
        return
    if args.index is None:
        raise SystemExit("--index is required unless --count is used")
    if args.index < 0 or args.index >= len(rows):
        raise SystemExit(f"index {args.index} is outside [0, {len(rows)})")
    row = rows[args.index]
    print(_train_tsv(row) if isinstance(row, TrainRow) else _eval_tsv(row))


if __name__ == "__main__":
    main()
