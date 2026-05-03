from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np


K_VALUES = [1, 2, 4, 8, 16, 32, 64]
BANDS = [
    ("band_train", "ID steps 1-10"),
    ("band_ood", "OOD steps 11-20"),
    ("band_hard_tail", "Hard tail steps 15-20"),
]
METRICS = [
    ("correct", "Correct pass@k"),
    ("valid", "Valid pass@k"),
    ("joint", "Joint pass@k"),
    ("valid_given_correct", "Valid given correct@k"),
    ("invalid_but_correct", "Invalid but correct@k"),
    ("format", "Format pass@k"),
    ("syntactic", "Syntactic pass@k"),
]

SCHEMAS = [
    "correct_plus_0p1_format",
    "correct_plus_valid_plus_0p1_format",
    "correct_times_valid_plus_0p1_format",
    "correct_plus_line_valid_plus_0p1_format",
    "correct_times_line_valid_plus_0p1_format",
]
SEEDS = [3407, 3408, 3409]

SCHEMA_LABELS = {
    "correct_plus_0p1_format": "correct + 0.1 format",
    "correct_plus_valid_plus_0p1_format": "correct + valid + 0.1 format",
    "correct_times_valid_plus_0p1_format": "correct * valid + 0.1 format",
    "correct_plus_line_valid_plus_0p1_format": "correct + line-valid + 0.1 format",
    "correct_times_line_valid_plus_0p1_format": "correct * line-valid + 0.1 format",
}

ABLATIONS = {
    "hard_v3_all_rewards": SCHEMAS,
    "hard_v3_no_validity_vs_binary_validity": [
        "correct_plus_0p1_format",
        "correct_plus_valid_plus_0p1_format",
        "correct_times_valid_plus_0p1_format",
    ],
    "hard_v3_no_validity_vs_line_validity": [
        "correct_plus_0p1_format",
        "correct_plus_line_valid_plus_0p1_format",
        "correct_times_line_valid_plus_0p1_format",
    ],
    "hard_v3_additive_validity": [
        "correct_plus_0p1_format",
        "correct_plus_valid_plus_0p1_format",
        "correct_plus_line_valid_plus_0p1_format",
    ],
    "hard_v3_multiplicative_validity": [
        "correct_plus_0p1_format",
        "correct_times_valid_plus_0p1_format",
        "correct_times_line_valid_plus_0p1_format",
    ],
    "hard_v3_binary_vs_line_validity": [
        "correct_plus_valid_plus_0p1_format",
        "correct_plus_line_valid_plus_0p1_format",
        "correct_times_valid_plus_0p1_format",
        "correct_times_line_valid_plus_0p1_format",
    ],
}


@dataclass(frozen=True)
class RunRecord:
    schema: str
    seed: int
    path: Path
    metrics: dict[str, float]


def metric_key(band: str, metric: str, k: int) -> str:
    if metric in {"valid_given_correct", "correct_given_valid", "invalid_but_correct", "valid_but_wrong"}:
        return f"synthetic_sampled/{band}/{metric}@{k}"
    return f"synthetic_sampled/{band}/{metric}_pass@{k}"


def load_records(in_dir: Path) -> list[RunRecord]:
    pattern = re.compile(r"reward_rl_hard_v3_(?P<schema>.+)_seed(?P<seed>\d+)_mrg_passk\.json$")
    records: list[RunRecord] = []
    for path in sorted(in_dir.glob("reward_rl_hard_v3_*_passk.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            RunRecord(
                schema=match.group("schema"),
                seed=int(match.group("seed")),
                path=path,
                metrics={k: float(v) for k, v in payload["metrics"].items() if isinstance(v, (int, float))},
            )
        )
    return records


def validate_records(records: list[RunRecord]) -> list[str]:
    problems: list[str] = []
    by_schema: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_schema[record.schema].append(record.seed)
        for band, _ in BANDS:
            for metric, _ in METRICS:
                for k in K_VALUES:
                    key = metric_key(band, metric, k)
                    if key not in record.metrics:
                        problems.append(f"missing {key} in {record.path.name}")
    for schema in SCHEMAS:
        seeds = sorted(by_schema.get(schema, []))
        if seeds != SEEDS:
            problems.append(f"{schema} has seeds {seeds}, expected {SEEDS}")
    extra = sorted(set(by_schema) - set(SCHEMAS))
    if extra:
        problems.append(f"unexpected schemas: {extra}")
    return problems


def aggregate(records: list[RunRecord], schema: str, band: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    schema_records = [record for record in records if record.schema == schema]
    vals_by_k: list[list[float]] = []
    for k in K_VALUES:
        key = metric_key(band, metric, k)
        vals_by_k.append([record.metrics[key] for record in schema_records if key in record.metrics])
    means = np.array([np.mean(vals) if vals else np.nan for vals in vals_by_k], dtype=float)
    stds = np.array([np.std(vals) if vals else np.nan for vals in vals_by_k], dtype=float)
    counts = np.array([len(vals) for vals in vals_by_k], dtype=int)
    return means, stds, counts


def write_tables(records: list[RunRecord], out_dir: Path) -> None:
    summary = ["schema\tband\tmetric\tpass@1_mean\tpass@1_std\tpass@8_mean\tpass@8_std\tpass@64_mean\tpass@64_std"]
    per_seed = ["schema\tseed\tband\tmetric\tpass@1\tpass@8\tpass@64"]
    for schema in SCHEMAS:
        for band, _ in BANDS:
            for metric, _ in METRICS:
                means, stds, _ = aggregate(records, schema, band, metric)
                summary.append(
                    f"{schema}\t{band}\t{metric}\t"
                    f"{means[0]:.6f}\t{stds[0]:.6f}\t{means[3]:.6f}\t{stds[3]:.6f}\t{means[-1]:.6f}\t{stds[-1]:.6f}"
                )
        for seed in SEEDS:
            record = next(r for r in records if r.schema == schema and r.seed == seed)
            for band, _ in BANDS:
                for metric, _ in METRICS:
                    per_seed.append(
                        f"{schema}\t{seed}\t{band}\t{metric}\t"
                        f"{record.metrics[metric_key(band, metric, 1)]:.6f}\t"
                        f"{record.metrics[metric_key(band, metric, 8)]:.6f}\t"
                        f"{record.metrics[metric_key(band, metric, 64)]:.6f}"
                    )

    (out_dir / "hard_v3_passk_summary.tsv").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out_dir / "hard_v3_passk_per_seed.tsv").write_text("\n".join(per_seed) + "\n", encoding="utf-8")
    manifest = {
        "input_records": [str(r.path) for r in records],
        "schemas": SCHEMAS,
        "schema_labels": SCHEMA_LABELS,
        "ablations": ABLATIONS,
        "bands": [band for band, _ in BANDS],
        "metrics": [metric for metric, _ in METRICS],
        "k_values": K_VALUES,
    }
    (out_dir / "hard_v3_passk_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def step_metric_key(metric: str, step: int) -> str:
    if metric == "format":
        metric = "format"
    return f"synthetic/step_{step}/{metric}"


def sampled_step_metric_key(metric: str, step: int, k: int) -> str:
    if metric in {"valid_given_correct", "correct_given_valid", "invalid_but_correct", "valid_but_wrong"}:
        return f"synthetic_sampled/step_{step}/{metric}@{k}"
    return f"synthetic_sampled/step_{step}/{metric}_pass@{k}"


def aggregate_step(records: list[RunRecord], schema: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    schema_records = [record for record in records if record.schema == schema]
    vals_by_step: list[list[float]] = []
    for step in range(1, 21):
        key = step_metric_key(metric, step)
        vals_by_step.append([record.metrics[key] for record in schema_records if key in record.metrics])
    means = np.array([np.mean(vals) if vals else np.nan for vals in vals_by_step], dtype=float)
    stds = np.array([np.std(vals) if vals else np.nan for vals in vals_by_step], dtype=float)
    return means, stds


def aggregate_sampled_step(records: list[RunRecord], schema: str, metric: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    schema_records = [record for record in records if record.schema == schema]
    vals_by_step: list[list[float]] = []
    for step in range(1, 21):
        key = sampled_step_metric_key(metric, step, k)
        vals_by_step.append([record.metrics[key] for record in schema_records if key in record.metrics])
    means = np.array([np.mean(vals) if vals else np.nan for vals in vals_by_step], dtype=float)
    stds = np.array([np.std(vals) if vals else np.nan for vals in vals_by_step], dtype=float)
    return means, stds


def write_step_tables(records: list[RunRecord], out_dir: Path) -> None:
    lines = ["schema\tstep\tmetric\tmean\tstd"]
    for schema in SCHEMAS:
        for step in range(1, 21):
            for metric in ["correct", "valid", "format", "syntactic"]:
                vals = [
                    record.metrics[step_metric_key(metric, step)]
                    for record in records
                    if record.schema == schema and step_metric_key(metric, step) in record.metrics
                ]
                lines.append(f"{schema}\t{step}\t{metric}\t{np.mean(vals):.6f}\t{np.std(vals):.6f}")
    (out_dir / "hard_v3_greedy_step_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sampled_lines = ["schema\tstep\tmetric\tk\tmean\tstd"]
    for schema in SCHEMAS:
        for step in range(1, 21):
            for metric in ["correct", "valid", "joint", "valid_given_correct", "invalid_but_correct"]:
                for k in [1, 8, 64]:
                    key = sampled_step_metric_key(metric, step, k)
                    vals = [record.metrics[key] for record in records if record.schema == schema and key in record.metrics]
                    sampled_lines.append(f"{schema}\t{step}\t{metric}\t{k}\t{np.mean(vals):.6f}\t{np.std(vals):.6f}")
    (out_dir / "hard_v3_sampled_step_summary.tsv").write_text("\n".join(sampled_lines) + "\n", encoding="utf-8")


def plot_step_curves(records: list[RunRecord], out_dir: Path) -> None:
    steps = np.arange(1, 21)
    for metric, label in [
        ("correct", "Greedy correct@1"),
        ("valid", "Greedy valid@1"),
        ("syntactic", "Greedy syntactic@1"),
        ("format", "Greedy format@1"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for schema in SCHEMAS:
            means, stds = aggregate_step(records, schema, metric)
            ax.plot(steps, means, marker="o", linewidth=2, label=SCHEMA_LABELS[schema])
            ax.fill_between(steps, means - stds, means + stds, alpha=0.12)
        ax.axvspan(10.5, 20.5, color="gray", alpha=0.08, label="OOD")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("required proof steps")
        ax.set_ylabel(label)
        ax.set_title(f"hard-v3: {label} by step")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"hard_v3_greedy_steps_{metric}.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig)

    for k in [1, 8, 64]:
        for metric, label in [
            ("correct", f"Correct pass@{k}"),
            ("valid", f"Valid pass@{k}"),
            ("joint", f"Joint pass@{k}"),
            ("valid_given_correct", f"Valid given correct@{k}"),
            ("invalid_but_correct", f"Invalid but correct@{k}"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for schema in SCHEMAS:
                means, stds = aggregate_sampled_step(records, schema, metric, k)
                ax.plot(steps, means, marker="o", linewidth=2, label=SCHEMA_LABELS[schema])
                ax.fill_between(steps, means - stds, means + stds, alpha=0.12)
            ax.axvspan(10.5, 20.5, color="gray", alpha=0.08, label="OOD")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(steps)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("required proof steps")
            ax.set_ylabel(label)
            ax.set_title(f"hard-v3: {label} by step")
            ax.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"hard_v3_sampled_steps_{metric}_k{k}.{ext}", bbox_inches="tight", dpi=180)
            plt.close(fig)


def plot_metric(records: list[RunRecord], out_dir: Path, ablation_name: str, schemas: list[str], metric: str, metric_label: str) -> None:
    fig, axes = plt.subplots(1, len(BANDS), figsize=(16, 4.8), sharey=True)
    for ax, (band, band_label) in zip(axes, BANDS, strict=True):
        for schema in schemas:
            means, stds, _ = aggregate(records, schema, band, metric)
            label = SCHEMA_LABELS[schema]
            ax.plot(K_VALUES, means, marker="o", linewidth=2, label=label)
            ax.fill_between(K_VALUES, means - stds, means + stds, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.set_title(band_label)
        ax.set_xlabel("k samples per prompt")
    axes[0].set_ylabel(metric_label)
    fig.suptitle(f"{ablation_name}: {metric_label}", y=1.02)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{ablation_name}_{metric}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_all(records: list[RunRecord], out_dir: Path) -> None:
    for ablation_name, schemas in ABLATIONS.items():
        for metric, metric_label in METRICS:
            plot_metric(records, out_dir, ablation_name, schemas, metric, metric_label)


def load_sample_rows(in_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"reward_rl_hard_v3_(?P<schema>.+)_seed(?P<seed>\d+)_mrg_samples\.jsonl$")
    for path in sorted(in_dir.glob("reward_rl_hard_v3_*_samples.jsonl")):
        match = pattern.match(path.name)
        if not match:
            continue
        for line_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_schema"] = match.group("schema")
            row["_seed"] = int(match.group("seed"))
            row["_path"] = path.name
            row["_line"] = line_idx
            rows.append(row)
    return rows


def write_sample_report(in_dir: Path, out_dir: Path) -> None:
    rows = load_sample_rows(in_dir)
    by_schema: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_schema[str(row["_schema"])].append(row)

    lines: list[str] = ["# Hard-v3 Sample Inspection", ""]
    lines.append(f"Loaded {len(rows)} representative sample rows from {len(list(in_dir.glob('*_samples.jsonl')))} files.")
    lines.append("")
    lines.append("## Aggregate Representative Samples")
    lines.append("")
    lines.append("schema\trows\tformat_mean\tsyntactic_mean\tvalid_mean\tcorrect_mean\tinvalid_correct_rows")
    for schema in SCHEMAS:
        schema_rows = by_schema[schema]
        if not schema_rows:
            continue
        means = {
            key: float(np.mean([float(r.get(key, 0.0)) for r in schema_rows]))
            for key in ["format_ok", "syntactic", "valid", "correct"]
        }
        invalid_correct = sum(1 for r in schema_rows if float(r.get("correct", 0.0)) > 0 and float(r.get("valid", 0.0)) <= 0)
        lines.append(
            f"{schema}\t{len(schema_rows)}\t{means['format_ok']:.3f}\t{means['syntactic']:.3f}\t"
            f"{means['valid']:.3f}\t{means['correct']:.3f}\t{invalid_correct}"
        )

    lines.append("")
    lines.append("## Representative Failure Rows")
    failure_rows = [
        r
        for r in rows
        if float(r.get("format_ok", 0.0)) < 1 or float(r.get("syntactic", 0.0)) < 1 or float(r.get("valid", 0.0)) < 1 or float(r.get("correct", 0.0)) < 1
    ]
    if not failure_rows:
        lines.append("")
        lines.append("No failures appear in the compact sample JSONL files. Use the aggregate pass@k metrics for real failure rates.")
    else:
        for row in failure_rows[:24]:
            generation = str(row.get("generation", "")).replace("\n", "\\n")
            prompt = str(row.get("prompt", "")).replace("\n", " ")
            lines.append("")
            lines.append(
                f"- schema={row['_schema']} seed={row['_seed']} step={row.get('step')} "
                f"format={row.get('format_ok')} syntactic={row.get('syntactic')} valid={row.get('valid')} correct={row.get('correct')}"
            )
            lines.append(f"  prompt: {shorten(prompt, width=220, placeholder=' ...')}")
            lines.append(f"  generation: {shorten(generation, width=800, placeholder=' ...')}")

    lines.append("")
    lines.append("## Clean Example")
    clean = next(
        (
            r
            for r in rows
            if str(r["_schema"]) == "correct_plus_0p1_format"
            and int(r["_seed"]) == 3407
            and float(r.get("valid", 0.0)) == 1.0
            and float(r.get("correct", 0.0)) == 1.0
        ),
        rows[0] if rows else None,
    )
    if clean:
        lines.append("")
        lines.append(f"schema={clean['_schema']} seed={clean['_seed']} step={clean.get('step')}")
        lines.append("")
        lines.append("```text")
        lines.append(str(clean.get("prompt", "")).rstrip())
        lines.append(str(clean.get("generation", "")).rstrip())
        lines.append("```")

    (out_dir / "hard_v3_sample_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/hard_v3")
    parser.add_argument("--out-dir", default="plots/hard_v3_passk")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(in_dir)
    problems = validate_records(records)
    if problems:
        (out_dir / "hard_v3_passk_validation_problems.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        raise SystemExit(f"Validation failed with {len(problems)} problems; see {out_dir / 'hard_v3_passk_validation_problems.txt'}")

    write_tables(records, out_dir)
    write_step_tables(records, out_dir)
    plot_all(records, out_dir)
    plot_step_curves(records, out_dir)
    write_sample_report(in_dir, out_dir)
    print(f"Wrote hard-v3 pass@k plots and reports to {out_dir}")


if __name__ == "__main__":
    main()
