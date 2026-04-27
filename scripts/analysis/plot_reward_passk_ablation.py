from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
    ("format", "Format pass@k"),
]


@dataclass(frozen=True)
class RunRecord:
    schema: str
    seed: int
    metrics: dict[str, float]


SCHEMA_LABELS = {
    "correct_plus_0p1_format": "correct + 0.1 format",
    "indicator_correct_and_format": "1[correct & format]",
    "correct_plus_valid_plus_0p1_format": "correct + 1.0 valid + 0.1 format",
    "correct_plus_0p75_valid_plus_0p1_format": "correct + 0.75 valid + 0.1 format",
    "correct_plus_0p5_valid_plus_0p1_format": "correct + 0.5 valid + 0.1 format",
    "correct_plus_0p25_valid_plus_0p1_format": "correct + 0.25 valid + 0.1 format",
    "correct_plus_line_valid_plus_0p1_format": "correct + line-valid + 0.1 format",
    "indicator_all": "1[correct & valid & format]",
}


def load_records(in_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    pattern = re.compile(r"reward_rl_reward_(?P<schema>.+)_seed(?P<seed>\d+)_mrg_passk\.json$")
    for path in sorted(in_dir.glob("reward_*_passk.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            RunRecord(
                schema=match.group("schema"),
                seed=int(match.group("seed")),
                metrics={k: float(v) for k, v in payload["metrics"].items() if isinstance(v, (int, float))},
            )
        )
    return records


def metric_key(band: str, metric: str, k: int) -> str:
    if metric == "valid_given_correct":
        return f"synthetic_sampled/{band}/valid_given_correct@{k}"
    return f"synthetic_sampled/{band}/{metric}_pass@{k}"


def aggregate(records: list[RunRecord], schema: str, band: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    vals_by_k: list[list[float]] = []
    schema_records = [record for record in records if record.schema == schema]
    for k in K_VALUES:
        vals = [record.metrics[metric_key(band, metric, k)] for record in schema_records if metric_key(band, metric, k) in record.metrics]
        vals_by_k.append(vals)
    means = np.array([np.mean(vals) if vals else np.nan for vals in vals_by_k], dtype=float)
    stds = np.array([np.std(vals) if vals else np.nan for vals in vals_by_k], dtype=float)
    return means, stds


def best_validity_scale_schema(records: list[RunRecord]) -> str:
    candidates = [
        "correct_plus_valid_plus_0p1_format",
        "correct_plus_0p75_valid_plus_0p1_format",
        "correct_plus_0p5_valid_plus_0p1_format",
        "correct_plus_0p25_valid_plus_0p1_format",
    ]
    scores: dict[str, float] = {}
    for schema in candidates:
        means, _ = aggregate(records, schema, "band_ood", "joint")
        scores[schema] = float(means[-1])
    return max(scores, key=scores.get)


def ablations(records: list[RunRecord]) -> dict[str, list[str]]:
    best_c = best_validity_scale_schema(records)
    return {
        "validity_scaling": [
            "correct_plus_0p1_format",
            "correct_plus_0p25_valid_plus_0p1_format",
            "correct_plus_0p5_valid_plus_0p1_format",
            "correct_plus_0p75_valid_plus_0p1_format",
            "correct_plus_valid_plus_0p1_format",
        ],
        "sparse_vs_dense_no_validity": [
            "indicator_correct_and_format",
            "correct_plus_0p1_format",
        ],
        "sparse_vs_dense_with_validity": [
            "indicator_all",
            "correct_plus_valid_plus_0p1_format",
        ],
        "sparse_validity_ablation": [
            "indicator_correct_and_format",
            "indicator_all",
        ],
        "no_validity_vs_best_validity_scale": [
            "correct_plus_0p1_format",
            best_c,
        ],
    }


def validate_records(records: list[RunRecord]) -> list[str]:
    problems: list[str] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[record.schema].append(record.seed)
        for band, _ in BANDS:
            for metric, _ in METRICS:
                for k in K_VALUES:
                    key = metric_key(band, metric, k)
                    if key not in record.metrics:
                        problems.append(f"missing {key} in {record.schema} seed {record.seed}")
    for schema, seeds in sorted(grouped.items()):
        if sorted(seeds) != [3407, 3408, 3409]:
            problems.append(f"{schema} has seeds {sorted(seeds)}, expected [3407, 3408, 3409]")
    return problems


def write_summary(records: list[RunRecord], out_dir: Path, ablation_map: dict[str, list[str]]) -> None:
    lines = ["schema\tband\tmetric\tpass@64_mean\tpass@64_std"]
    for schema in sorted({r.schema for r in records}):
        for band, _ in BANDS:
            for metric, _ in METRICS:
                means, stds = aggregate(records, schema, band, metric)
                lines.append(f"{schema}\t{band}\t{metric}\t{means[-1]:.6f}\t{stds[-1]:.6f}")
    (out_dir / "reward_passk_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = best_validity_scale_schema(records)
    payload = {
        "best_validity_scale_schema_by_ood_joint_pass@64": best,
        "ablations": ablation_map,
        "k_values": K_VALUES,
        "bands": [band for band, _ in BANDS],
        "metrics": [metric for metric, _ in METRICS],
    }
    (out_dir / "reward_passk_plot_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_metric(records: list[RunRecord], out_dir: Path, ablation_name: str, schemas: list[str], metric: str, metric_label: str) -> None:
    fig, axes = plt.subplots(1, len(BANDS), figsize=(16, 4.8), sharey=True)
    for ax, (band, band_label) in zip(axes, BANDS, strict=True):
        for schema in schemas:
            means, stds = aggregate(records, schema, band, metric)
            label = SCHEMA_LABELS.get(schema, schema)
            ax.plot(K_VALUES, means, marker="o", linewidth=2, label=label)
            ax.fill_between(K_VALUES, means - stds, means + stds, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.set_title(band_label)
        ax.set_xlabel("k completed samples per prompt")
    axes[0].set_ylabel(metric_label)
    fig.suptitle(f"{ablation_name}: {metric_label}", y=1.02)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{ablation_name}_{metric}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval")
    parser.add_argument("--out-dir", default="plots/reward_passk")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(in_dir)
    problems = validate_records(records)
    if problems:
        (out_dir / "reward_passk_validation_problems.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        raise SystemExit(f"Validation failed with {len(problems)} problems; see {out_dir / 'reward_passk_validation_problems.txt'}")

    ablation_map = ablations(records)
    write_summary(records, out_dir, ablation_map)
    for ablation_name, schemas in ablation_map.items():
        for metric, metric_label in METRICS:
            plot_metric(records, out_dir, ablation_name, schemas, metric, metric_label)
    print(f"Wrote plots and summaries to {out_dir}")


if __name__ == "__main__":
    main()
