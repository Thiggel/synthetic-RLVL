from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


FINAL_RE = re.compile(
    r"sft_(?:hfsa_depth_scaling|branchproof_unique_v2)_(?P<template>logic|nl_exact)_"
    r"train1to(?P<train>\d+)_10k_seed(?P<seed>\d+)_passk\.json$"
)
INTERMEDIATE_RE = re.compile(
    r"sft_hfsa_depth_scaling_(?P<template>logic|nl_exact)_train1to(?P<train>\d+)_10k_seed(?P<seed>\d+)_"
    r"checkpoint-(?P<ckpt>\d+)_passk\.json$"
)
DEPTHS = [1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
INTERMEDIATE_DEPTHS = [1, 5, 10, 15, 20, 25, 30, 40, 50]
K_VALUES = (1, 2, 4, 8, 16)
K = 16
THRESHOLDS = [0.8, 0.5, 0.25]
TEMPLATE_LABELS = {"logic": "Logic", "nl_exact": "Natural language"}
COLORS = {"logic": "#1f77b4", "nl_exact": "#d62728"}
MARKERS = {5: "o", 10: "s", 15: "^", 20: "D", 25: "P"}


@dataclass(frozen=True)
class RunRecord:
    path: Path
    template: str
    train_max: int
    seed: int
    checkpoint_step: int | None
    elapsed_seconds: float
    metrics: dict[str, float]


def joint_metric(template: str) -> str:
    return "citation_free_joint_pass" if template == "logic" else "nl_logic_joint_pass"


def valid_metric(template: str) -> str:
    return "citation_free_valid_pass" if template == "logic" else "nl_logic_citation_free_valid_pass"


def valid_given_correct_metric(template: str) -> str:
    return "citation_free_valid_given_correct" if template == "logic" else "nl_logic_valid_given_correct"


def load_records(directory: Path, pattern: re.Pattern[str], intermediate: bool) -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in sorted(directory.glob("*_passk.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = {k: float(v) for k, v in payload.get("metrics", {}).items() if isinstance(v, (int, float))}
        records.append(
            RunRecord(
                path=path,
                template=match.group("template"),
                train_max=int(match.group("train")),
                seed=int(match.group("seed")),
                checkpoint_step=int(match.group("ckpt")) if intermediate else None,
                elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
                metrics=metrics,
            )
        )
    return records


def metric(record: RunRecord, suffix: str) -> float | None:
    return record.metrics.get(f"synthetic_sampled/{suffix}")


def step_metric(record: RunRecord, depth: int, metric_name: str, k: int = K) -> float | None:
    return metric(record, f"step_{depth}/{metric_name}@{k}")


def band_metric(record: RunRecord, band: str, metric_name: str, k: int = K) -> float | None:
    return metric(record, f"band_{band}/{metric_name}@{k}")


def greedy_step_metric(record: RunRecord, depth: int, metric_name: str) -> float | None:
    return record.metrics.get(f"synthetic/step_{depth}/{metric_name}")


def greedy_band_metric(record: RunRecord, band: str, metric_name: str) -> float | None:
    predicates = {
        "train": lambda depth: depth <= record.train_max,
        "ood": lambda depth: depth > record.train_max,
        "hard_tail": lambda depth: depth >= 15,
    }
    values = [
        value
        for depth in DEPTHS
        if predicates[band](depth)
        and (value := greedy_step_metric(record, depth, metric_name)) is not None
    ]
    return mean(values) if values else None


def row_metric(record: RunRecord, band: str, metric_name: str, k: int = K) -> float:
    value = band_metric(record, band, metric_name, k)
    return float("nan") if value is None else value


def stats(values: list[float]) -> tuple[float, float, int]:
    clean = [v for v in values if v == v]
    if not clean:
        return float("nan"), float("nan"), 0
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0, len(clean)


def trapezoid_auc(points: list[tuple[int, float]]) -> float:
    clean = sorted((x, y) for x, y in points if y == y)
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0][1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(clean, clean[1:], strict=False):
        area += (x1 - x0) * (y0 + y1) / 2.0
    span = clean[-1][0] - clean[0][0]
    return area / span if span else clean[0][1]


def max_depth_at_threshold(points: list[tuple[int, float]], threshold: float) -> int:
    depths = [depth for depth, value in points if value == value and value >= threshold]
    return max(depths) if depths else 0


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_compact_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| template | train | OOD c@16 | OOD joint@16 | OOD joint AUC | depth50 joint@16 | max joint>=0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (str(r["template"]), int(r["train_max"]))):
        lines.append(
            "| `{template}` | {train} | {ood_c:.3f} | {ood_j:.3f} | {auc:.3f} | {d50:.3f} | {max_depth} |".format(
                template=row["template"],
                train=int(row["train_max"]),
                ood_c=float(row["ood_correct16_mean"]),
                ood_j=float(row["ood_joint16_mean"]),
                auc=float(row["auc_ood_joint16"]),
                d50=float(row["depth50_joint16_mean"]),
                max_depth=int(row["max_depth_joint16_ge_0p5"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_primary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| template | train | greedy OOD c | OOD c@1 | OOD joint@1 | OOD c@4 | OOD c@8 | OOD c@16 | OOD joint@16 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: (str(r["template"]), int(r["train_max"]))):
        lines.append(
            "| `{template}` | {train} | {greedy:.3f} | {c1:.3f} | {j1:.3f} | {c4:.3f} | {c8:.3f} | {c16:.3f} | {j16:.3f} |".format(
                template=row["template"],
                train=int(row["train_max"]),
                greedy=float(row["greedy_ood_correct_mean"]),
                c1=float(row["ood_correct1_mean"]),
                j1=float(row["ood_joint1_mean"]),
                c4=float(row["ood_correct4_mean"]),
                c8=float(row["ood_correct8_mean"]),
                c16=float(row["ood_correct16_mean"]),
                j16=float(row["ood_joint16_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_run_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in sorted(records, key=lambda r: (r.template, r.train_max, r.seed)):
        joint = joint_metric(record.template)
        valid = valid_metric(record.template)
        vgc = valid_given_correct_metric(record.template)
        greedy_valid = "citation_free_valid" if record.template == "logic" else "nl_logic_citation_free_valid"
        row: dict[str, object] = {
            "template": record.template,
            "train_max": record.train_max,
            "seed": record.seed,
            "elapsed_hours": record.elapsed_seconds / 3600.0,
            "ood_valid_given_correct16": row_metric(record, "ood", vgc),
            "greedy_train_correct": greedy_band_metric(record, "train", "correct"),
            "greedy_train_valid": greedy_band_metric(record, "train", greedy_valid),
            "greedy_ood_correct": greedy_band_metric(record, "ood", "correct"),
            "greedy_ood_valid": greedy_band_metric(record, "ood", greedy_valid),
            "greedy_hard_correct": greedy_band_metric(record, "hard_tail", "correct"),
            "greedy_hard_valid": greedy_band_metric(record, "hard_tail", greedy_valid),
            "greedy_depth50_correct": greedy_step_metric(record, 50, "correct"),
            "greedy_depth50_valid": greedy_step_metric(record, 50, greedy_valid),
        }
        for k in K_VALUES:
            row.update(
                {
                    f"train_correct{k}": row_metric(record, "train", "correct_pass", k),
                    f"train_valid{k}": row_metric(record, "train", valid, k),
                    f"train_joint{k}": row_metric(record, "train", joint, k),
                    f"ood_correct{k}": row_metric(record, "ood", "correct_pass", k),
                    f"ood_valid{k}": row_metric(record, "ood", valid, k),
                    f"ood_joint{k}": row_metric(record, "ood", joint, k),
                    f"hard_correct{k}": row_metric(record, "hard_tail", "correct_pass", k),
                    f"hard_valid{k}": row_metric(record, "hard_tail", valid, k),
                    f"hard_joint{k}": row_metric(record, "hard_tail", joint, k),
                    f"depth50_correct{k}": step_metric(record, 50, "correct_pass", k),
                    f"depth50_valid{k}": step_metric(record, 50, valid, k),
                    f"depth50_joint{k}": step_metric(record, 50, joint, k),
                }
            )
        rows.append(row)
    return rows


def final_depth_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in sorted(records, key=lambda r: (r.template, r.train_max, r.seed)):
        joint = joint_metric(record.template)
        valid = valid_metric(record.template)
        for depth in DEPTHS:
            row: dict[str, object] = {
                "template": record.template,
                "train_max": record.train_max,
                "seed": record.seed,
                "depth": depth,
                "in_train": int(depth <= record.train_max),
                "greedy_correct": greedy_step_metric(record, depth, "correct"),
                "greedy_valid": greedy_step_metric(
                    record,
                    depth,
                    "citation_free_valid"
                    if record.template == "logic"
                    else "nl_logic_citation_free_valid",
                ),
            }
            for k in K_VALUES:
                row[f"correct{k}"] = step_metric(record, depth, "correct_pass", k)
                row[f"valid{k}"] = step_metric(record, depth, valid, k)
                row[f"joint{k}"] = step_metric(record, depth, joint, k)
            rows.append(row)
    return rows


def group_summary_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.template, record.train_max)].append(record)
    rows: list[dict[str, object]] = []
    for (template, train_max), items in sorted(grouped.items()):
        joint = joint_metric(template)
        greedy_valid = "citation_free_valid" if template == "logic" else "nl_logic_citation_free_valid"
        values: dict[str, list[float | None]] = {
            "greedy_train_correct": [greedy_band_metric(r, "train", "correct") for r in items],
            "greedy_train_valid": [greedy_band_metric(r, "train", greedy_valid) for r in items],
            "greedy_ood_correct": [greedy_band_metric(r, "ood", "correct") for r in items],
            "greedy_ood_valid": [greedy_band_metric(r, "ood", greedy_valid) for r in items],
            "greedy_hard_correct": [greedy_band_metric(r, "hard_tail", "correct") for r in items],
            "greedy_hard_valid": [greedy_band_metric(r, "hard_tail", greedy_valid) for r in items],
            "greedy_depth50_correct": [greedy_step_metric(r, 50, "correct") for r in items],
            "greedy_depth50_valid": [greedy_step_metric(r, 50, greedy_valid) for r in items],
            "elapsed_hours": [r.elapsed_seconds / 3600.0 for r in items],
        }
        for k in K_VALUES:
            values.update(
                {
                    f"train_correct{k}": [row_metric(r, "train", "correct_pass", k) for r in items],
                    f"train_valid{k}": [row_metric(r, "train", valid_metric(template), k) for r in items],
                    f"train_joint{k}": [row_metric(r, "train", joint, k) for r in items],
                    f"ood_correct{k}": [row_metric(r, "ood", "correct_pass", k) for r in items],
                    f"ood_valid{k}": [row_metric(r, "ood", valid_metric(template), k) for r in items],
                    f"ood_joint{k}": [row_metric(r, "ood", joint, k) for r in items],
                    f"hard_correct{k}": [row_metric(r, "hard_tail", "correct_pass", k) for r in items],
                    f"hard_valid{k}": [row_metric(r, "hard_tail", valid_metric(template), k) for r in items],
                    f"hard_joint{k}": [row_metric(r, "hard_tail", joint, k) for r in items],
                    f"depth50_correct{k}": [step_metric(r, 50, "correct_pass", k) for r in items],
                    f"depth50_valid{k}": [step_metric(r, 50, valid_metric(template), k) for r in items],
                    f"depth50_joint{k}": [step_metric(r, 50, joint, k) for r in items],
                }
            )
        row: dict[str, object] = {"template": template, "train_max": train_max, "n": len(items)}
        for name, vals in values.items():
            avg, std, _ = stats([float(v) for v in vals if v is not None])
            row[f"{name}_mean"] = avg
            row[f"{name}_std"] = std

        for metric_name, output_name in [("correct_pass", "correct16"), (joint, "joint16")]:
            all_points: list[tuple[int, float]] = []
            ood_points: list[tuple[int, float]] = []
            for depth in DEPTHS:
                vals = [step_metric(r, depth, metric_name) for r in items]
                avg, _, n = stats([float(v) for v in vals if v is not None])
                if n:
                    all_points.append((depth, avg))
                    if depth > train_max:
                        ood_points.append((depth, avg))
            row[f"auc_all_{output_name}"] = trapezoid_auc(all_points)
            row[f"auc_ood_{output_name}"] = trapezoid_auc(ood_points)
            for threshold in THRESHOLDS:
                suffix = str(threshold).replace(".", "p")
                row[f"max_depth_{output_name}_ge_{suffix}"] = max_depth_at_threshold(all_points, threshold)
        rows.append(row)
    return rows


def grouped_depth_summary_rows(depth_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in depth_rows:
        grouped[(str(row["template"]), int(row["train_max"]), int(row["depth"]))].append(row)
    rows: list[dict[str, object]] = []
    for (template, train_max, depth), items in sorted(grouped.items()):
        out: dict[str, object] = {"template": template, "train_max": train_max, "depth": depth, "n": len(items)}
        metric_names = ["greedy_correct", "greedy_valid"] + [
            f"{metric_name}{k}"
            for k in K_VALUES
            for metric_name in ("correct", "valid", "joint")
        ]
        for name in metric_names:
            vals = [float(item[name]) for item in items if item[name] not in (None, "")]
            avg, std, _ = stats(vals)
            out[f"{name}_mean"] = avg
            out[f"{name}_std"] = std
        rows.append(out)
    return rows


def paired_delta_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    by_key = {(r.template, r.train_max, r.seed): r for r in records}
    rows: list[dict[str, object]] = []
    for train_max in sorted({r.train_max for r in records}):
        for seed in sorted({r.seed for r in records}):
            logic = by_key.get(("logic", train_max, seed))
            nl = by_key.get(("nl_exact", train_max, seed))
            if not logic or not nl:
                continue
            logic_joint = joint_metric("logic")
            nl_joint = joint_metric("nl_exact")
            row: dict[str, object] = {
                "train_max": train_max,
                "seed": seed,
                "delta_greedy_ood_correct": float(
                    greedy_band_metric(logic, "ood", "correct") or 0.0
                )
                - float(greedy_band_metric(nl, "ood", "correct") or 0.0),
                "delta_greedy_depth50_correct": float(
                    greedy_step_metric(logic, 50, "correct") or 0.0
                )
                - float(greedy_step_metric(nl, 50, "correct") or 0.0),
            }
            for k in K_VALUES:
                row.update(
                    {
                        f"delta_ood_correct{k}": row_metric(logic, "ood", "correct_pass", k)
                        - row_metric(nl, "ood", "correct_pass", k),
                        f"delta_ood_joint{k}": row_metric(logic, "ood", logic_joint, k)
                        - row_metric(nl, "ood", nl_joint, k),
                        f"delta_depth50_correct{k}": float(
                            step_metric(logic, 50, "correct_pass", k) or 0.0
                        )
                        - float(step_metric(nl, 50, "correct_pass", k) or 0.0),
                        f"delta_depth50_joint{k}": float(
                            step_metric(logic, 50, logic_joint, k) or 0.0
                        )
                        - float(step_metric(nl, 50, nl_joint, k) or 0.0),
                    }
                )
            rows.append(row)
    return rows


def intermediate_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in sorted(records, key=lambda r: (r.template, r.train_max, r.checkpoint_step or 0, r.seed)):
        joint = joint_metric(record.template)
        rows.append(
            {
                "template": record.template,
                "train_max": record.train_max,
                "seed": record.seed,
                "checkpoint_step": record.checkpoint_step,
                "elapsed_hours": record.elapsed_seconds / 3600.0,
                "train_correct16": row_metric(record, "train", "correct_pass"),
                "train_joint16": row_metric(record, "train", joint),
                "ood_correct1": row_metric(record, "ood", "correct_pass", 1),
                "ood_correct16": row_metric(record, "ood", "correct_pass"),
                "ood_joint16": row_metric(record, "ood", joint),
                "hard_joint16": row_metric(record, "hard_tail", joint),
                "depth50_correct16": step_metric(record, 50, "correct_pass"),
                "depth50_joint16": step_metric(record, 50, joint),
            }
        )
    return rows


def final_records_complete(records: list[RunRecord], *, strict_metrics: bool = False) -> list[str]:
    problems: list[str] = []
    expected = {(template, train, seed) for template in ("logic", "nl_exact") for train in (5, 10, 15, 20, 25) for seed in (3407, 3408, 3409)}
    observed_keys = [(r.template, r.train_max, r.seed) for r in records]
    observed = set(observed_keys)
    for missing in sorted(expected - observed):
        problems.append(f"missing final {missing}")
    duplicates = sorted(key for key in observed if observed_keys.count(key) > 1)
    for duplicate in duplicates:
        problems.append(f"duplicate final {duplicate}")
    if not strict_metrics:
        return problems

    for record in records:
        run_id = (record.template, record.train_max, record.seed)
        expected_prompts = len(DEPTHS) * 32
        if record.metrics.get("posthoc/prompts") != expected_prompts:
            problems.append(
                f"{run_id} posthoc/prompts={record.metrics.get('posthoc/prompts')}, expected {expected_prompts}"
            )
        if record.metrics.get("posthoc/sampled_generations_per_prompt") != 16:
            problems.append(
                f"{run_id} sampled_generations={record.metrics.get('posthoc/sampled_generations_per_prompt')}, expected 16"
            )
        sampled_metric_names = ("correct_pass", valid_metric(record.template), joint_metric(record.template))
        greedy_valid_metric = "citation_free_valid" if record.template == "logic" else "nl_logic_citation_free_valid"
        for depth in DEPTHS:
            for metric_name in ("correct", greedy_valid_metric):
                key = f"synthetic/step_{depth}/{metric_name}"
                value = record.metrics.get(key)
                if value is None or not 0.0 <= value <= 1.0:
                    problems.append(f"{run_id} missing or invalid metric {key}")
            for metric_name in sampled_metric_names:
                previous: float | None = None
                for k in K_VALUES:
                    key = f"synthetic_sampled/step_{depth}/{metric_name}@{k}"
                    value = record.metrics.get(key)
                    if value is None or not 0.0 <= value <= 1.0:
                        problems.append(f"{run_id} missing or invalid metric {key}")
                        continue
                    if previous is not None and value + 1e-9 < previous:
                        problems.append(f"{run_id} non-monotonic metric {key}={value} < {previous}")
                    previous = value
        for band in ("train", "ood", "hard_tail"):
            for metric_name in sampled_metric_names:
                previous = None
                for k in K_VALUES:
                    key = f"synthetic_sampled/band_{band}/{metric_name}@{k}"
                    value = record.metrics.get(key)
                    if value is None or not 0.0 <= value <= 1.0:
                        problems.append(f"{run_id} missing or invalid metric {key}")
                        continue
                    if previous is not None and value + 1e-9 < previous:
                        problems.append(f"{run_id} non-monotonic metric {key}={value} < {previous}")
                    previous = value
    return problems


def make_plots(out_dir: Path, group_rows: list[dict[str, object]], depth_group_rows: list[dict[str, object]], deltas: list[dict[str, object]], intermediate: list[dict[str, object]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    for template in ("logic", "nl_exact"):
        rows = sorted(
            (row for row in group_rows if row["template"] == template),
            key=lambda row: int(row["train_max"]),
        )
        xs = [int(row["train_max"]) for row in rows]
        for ax, metric_name, title in (
            (axes[0], "greedy_ood_correct", "Greedy OOD correctness"),
            (axes[1], "ood_correct1", "Sampled pass@1 OOD correctness"),
        ):
            ax.errorbar(
                xs,
                [float(row[f"{metric_name}_mean"]) for row in rows],
                yerr=[float(row[f"{metric_name}_std"]) for row in rows],
                marker="o",
                linewidth=2,
                capsize=3,
                label=TEMPLATE_LABELS[template],
                color=COLORS[template],
            )
            ax.set_title(title)
            ax.set_xlabel("max train depth")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            fig_dir / f"final_primary_ood_correctness.{ext}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    for template in ("logic", "nl_exact"):
        row = next(
            item
            for item in group_rows
            if item["template"] == template and int(item["train_max"]) == 25
        )
        for ax, metric_name, title in (
            (axes[0], "ood_correct", "OOD correctness versus sample budget"),
            (axes[1], "ood_joint", "OOD correct and valid versus sample budget"),
        ):
            ax.errorbar(
                K_VALUES,
                [float(row[f"{metric_name}{k}_mean"]) for k in K_VALUES],
                yerr=[float(row[f"{metric_name}{k}_std"]) for k in K_VALUES],
                marker="o",
                linewidth=2,
                capsize=3,
                label=TEMPLATE_LABELS[template],
                color=COLORS[template],
            )
            ax.set_title(title)
            ax.set_xlabel("samples k")
            ax.set_xticks(K_VALUES)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("pass@k")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            fig_dir / f"final_train25_sampling_efficiency.{ext}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for template in ("logic", "nl_exact"):
        rows = [r for r in group_rows if r["template"] == template]
        rows.sort(key=lambda r: int(r["train_max"]))
        xs = [int(r["train_max"]) for r in rows]
        for ax, metric_name, label in [
            (axes[0], "ood_correct16", "OOD correct@16"),
            (axes[1], "ood_joint16", "OOD joint@16"),
        ]:
            ys = [float(r[f"{metric_name}_mean"]) for r in rows]
            yerr = [float(r[f"{metric_name}_std"]) for r in rows]
            ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=3, label=TEMPLATE_LABELS[template], color=COLORS[template])
            ax.set_title(label)
            ax.set_xlabel("max train depth")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("pass@16")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"final_ood_metrics_by_train.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for metric_name, title, output in [
        ("greedy_correct", "Greedy correctness by eval depth", "final_depth_greedy_correct"),
        ("correct1", "Sampled pass@1 correctness by eval depth", "final_depth_correct1"),
        ("joint1", "Sampled pass@1 correct and valid by eval depth", "final_depth_joint1"),
        ("correct16", "Correct@16 by eval depth", "final_depth_correct16"),
        ("joint16", "Joint valid+correct@16 by eval depth", "final_depth_joint16"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, template in zip(axes, ("logic", "nl_exact"), strict=True):
            for train in (5, 10, 15, 20, 25):
                rows = [
                    r
                    for r in depth_group_rows
                    if r["template"] == template and int(r["train_max"]) == train
                ]
                rows.sort(key=lambda r: int(r["depth"]))
                if not rows:
                    continue
                ax.plot(
                    [int(r["depth"]) for r in rows],
                    [float(r[f"{metric_name}_mean"]) for r in rows],
                    marker=MARKERS[train],
                    linewidth=2,
                    label=f"train 1..{train}",
                )
            ax.axvline(25, color="#666666", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_title(TEMPLATE_LABELS[template])
            ax.set_xlabel("eval depth")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel(title)
        axes[1].legend(loc="lower left", fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{output}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if deltas:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in deltas:
            grouped[int(row["train_max"])].append(float(row["delta_ood_joint16"]))
        xs = sorted(grouped)
        means = [mean(grouped[x]) for x in xs]
        stds = [pstdev(grouped[x]) if len(grouped[x]) > 1 else 0.0 for x in xs]
        ax.axhline(0.0, color="#444444", linewidth=1)
        ax.bar([str(x) for x in xs], means, yerr=stds, color="#2ca02c", alpha=0.8, capsize=3)
        ax.set_xlabel("max train depth")
        ax.set_ylabel("logic - NL OOD joint@16")
        ax.set_title("Paired seed deltas")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"paired_delta_ood_joint16.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if intermediate:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, metric_name, ylabel in [
            (axes[0], "ood_correct16", "OOD correct@16"),
            (axes[1], "ood_joint16", "OOD joint@16"),
        ]:
            for template in ("logic", "nl_exact"):
                for train in (5, 10, 15, 20, 25):
                    rows = [
                        r
                        for r in intermediate
                        if r["template"] == template and int(r["train_max"]) == train
                    ]
                    rows.sort(key=lambda r: int(r["checkpoint_step"]))
                    if not rows:
                        continue
                    style = "-" if template == "logic" else "--"
                    ax.plot(
                        [int(r["checkpoint_step"]) for r in rows],
                        [float(r[metric_name]) for r in rows],
                        linestyle=style,
                        marker=MARKERS[train],
                        linewidth=1.8,
                        color=COLORS[template],
                        alpha=0.55 + train / 80,
                        label=f"{TEMPLATE_LABELS[template]} 1..{train}",
                    )
            ax.set_xlabel("checkpoint step")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
        fig.suptitle("Seed 3407 intermediate checkpoint curves")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"intermediate_seed3407_curves.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default=None)
    parser.add_argument("--intermediate-dir", default=None)
    parser.add_argument("--skip-intermediate", action="store_true")
    parser.add_argument("--strict-final-grid", action="store_true")
    parser.add_argument("--out-dir", default="analysis/hfsa_depth_scaling_2026-05-23")
    args = parser.parse_args()

    work = Path(__file__).resolve().parents[2]
    final_dir = Path(args.final_dir) if args.final_dir else Path("/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/hfsa_depth_scaling_sparse")
    intermediate_dir = None
    if not args.skip_intermediate:
        intermediate_dir = (
            Path(args.intermediate_dir)
            if args.intermediate_dir
            else Path("/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/hfsa_depth_scaling_intermediate_sparse")
        )
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = work / out_dir
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    final_records = load_records(final_dir, FINAL_RE, intermediate=False)
    intermediate_records = (
        load_records(intermediate_dir, INTERMEDIATE_RE, intermediate=True)
        if intermediate_dir is not None
        else []
    )
    problems = final_records_complete(final_records, strict_metrics=args.strict_final_grid)
    manifest = {
        "final_dir": str(final_dir),
        "intermediate_dir": str(intermediate_dir) if intermediate_dir is not None else None,
        "final_json_count": len(final_records),
        "intermediate_json_count": len(intermediate_records),
        "depths": DEPTHS,
        "intermediate_depths": INTERMEDIATE_DEPTHS,
        "metric_policy": {
            "logic_joint": "citation_free_joint_pass",
            "nl_exact_joint": "nl_logic_joint_pass",
        },
        "strict_final_grid": args.strict_final_grid,
        "problems": problems,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if problems:
        raise SystemExit("Final sparse grid is incomplete; see manifest.json")

    final_runs = final_run_rows(final_records)
    final_depth = final_depth_rows(final_records)
    group_summary = group_summary_rows(final_records)
    depth_summary = grouped_depth_summary_rows(final_depth)
    deltas = paired_delta_rows(final_records)
    inter = intermediate_rows(intermediate_records)

    write_csv(
        tables / "final_run_metrics.csv",
        final_runs,
        list(final_runs[0].keys()),
    )
    write_csv(
        tables / "final_group_summary.csv",
        group_summary,
        list(group_summary[0].keys()),
    )
    write_compact_markdown(tables / "final_group_summary_compact.md", group_summary)
    write_primary_markdown(tables / "final_primary_summary.md", group_summary)
    write_csv(
        tables / "final_depth_curves.csv",
        final_depth,
        list(final_depth[0].keys()),
    )
    write_csv(
        tables / "final_depth_group_summary.csv",
        depth_summary,
        list(depth_summary[0].keys()),
    )
    write_csv(
        tables / "paired_seed_deltas.csv",
        deltas,
        list(deltas[0].keys()),
    )
    if inter:
        write_csv(
            tables / "intermediate_seed3407_metrics.csv",
            inter,
            list(inter[0].keys()),
        )
    make_plots(out_dir, group_summary, depth_summary, deltas, inter)
    print(f"Wrote HFSA depth-scaling analysis to {out_dir}")


if __name__ == "__main__":
    main()
