#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from hfsa_validity_analysis_lib import classify_generation, iter_metric_pairs_from_log_line, parse_run_name, strip_ansi

PASSK_RE = re.compile(r"synthetic_sampled/step_(?P<step>\d+)/(?P<metric>[^@]+)@(?P<k>\d+)$")
GREEDY_RE = re.compile(r"synthetic/step_(?P<step>\d+)/(?P<metric>[^/]+)$")
LOG_RUN_RE = re.compile(r"rl_hfsa_easy500_train(?:0p0|0p5)_[A-Za-z0-9_]+_seed\d+_mrg")

LABELS = {
    "train0p0_correct_only": "shortcut0.0 / correct only",
    "train0p0_validity_gated": "shortcut0.0 / correct & cf-valid",
    "train0p5_correct_only": "shortcut0.5 / correct only",
    "train0p5_validity_gated": "shortcut0.5 / correct & cf-valid",
}
COLORS = {
    "train0p0_correct_only": "#345995",
    "train0p0_validity_gated": "#03CEA4",
    "train0p5_correct_only": "#FB4D3D",
    "train0p5_validity_gated": "#CA1551",
}


def load_passk(passk_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_rows: list[dict[str, Any]] = []
    greedy_rows: list[dict[str, Any]] = []
    for path in sorted(passk_dir.glob("*_passk.json")):
        info = parse_run_name(path.name)
        obj = json.loads(path.read_text())
        metrics = obj.get("metrics", {})
        for key, val in metrics.items():
            m = PASSK_RE.match(key)
            if m:
                sampled_rows.append({**info, "step": int(m.group("step")), "metric": m.group("metric"), "k": int(m.group("k")), "value": float(val)})
                continue
            g = GREEDY_RE.match(key)
            if g:
                greedy_rows.append({**info, "step": int(g.group("step")), "metric": g.group("metric"), "value": float(val)})
    return pd.DataFrame(sampled_rows), pd.DataFrame(greedy_rows)


def classify_existing_samples(passk_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(passk_dir.glob("*_samples.jsonl")):
        info = parse_run_name(path.name)
        with path.open(encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                row = json.loads(line)
                cls = classify_generation(
                    generation=row.get("generation", ""),
                    prompt=row.get("prompt", ""),
                    step=int(row.get("step", 0)),
                    gold_answer=str(row.get("gold_answer", "")),
                    shortcut_answer=row.get("shortcut_answer"),
                    metadata=row.get("metadata") or {},
                    source="passk_greedy_sample",
                )
                rows.append({**info, "sample_path": str(path), "sample_line": line_idx, **{k: v for k, v in cls.items() if k not in {"proof_formulas", "atom_texts"}}})
    return pd.DataFrame(rows)


def load_targeted(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("generation"):
                    cls = classify_generation(
                        generation=row.get("generation", ""),
                        prompt=row.get("prompt", ""),
                        step=int(row.get("step", 0)),
                        gold_answer=str(row.get("gold_answer", "")),
                        shortcut_answer=row.get("shortcut_answer"),
                        metadata=row.get("metadata") or {},
                        source="targeted_probe",
                    )
                    row.update({k: v for k, v in cls.items() if k not in {"proof_formulas", "atom_texts"}})
                row["targeted_path"] = str(path)
                rows.append(row)
    return pd.DataFrame(rows)


def parse_training_logs(log_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("posttrain_hfsa_easy500_*.out")):
        text = path.read_text(errors="ignore")
        matches = LOG_RUN_RE.findall(text)
        run_name = matches[-1] if matches else path.stem
        info = parse_run_name(run_name)
        for line in text.splitlines():
            clean = strip_ansi(line)
            if "step:" not in clean or "train_aux/reward" not in clean:
                continue
            metrics = dict(iter_metric_pairs_from_log_line(clean))
            if not metrics:
                continue
            step = metrics.get("training/global_step") or metrics.get("step")
            if step is None:
                m = re.search(r"(?:^|\s)step:(\d+)\s+-", clean)
                step = float(m.group(1)) if m else math.nan
            row = {**info, "log_path": str(path), "step": int(step) if not math.isnan(float(step)) else -1}
            for key in [
                "critic/score/mean",
                "critic/score/max",
                "critic/score/min",
                "train_aux/reward/format/mean",
                "train_aux/reward/correct/mean",
                "train_aux/reward/citation_free_valid/mean",
                "train_aux/reward/citation_free_line_valid/mean",
                "response_length/mean",
                "actor/entropy",
                "actor/pg_loss",
            ]:
                if key in metrics:
                    row[key] = metrics[key]
            # With format near 1.0, score - 0.1 is P(correct) for correct-only and
            # P(correct & citation-free-valid) for indicator-gated runs.
            if "critic/score/mean" in row:
                row["derived_main_reward_event"] = max(0.0, min(1.0, float(row["critic/score/mean"]) - 0.1))
            if "critic/score/max" in row and "critic/score/min" in row:
                row["critic/score/range"] = float(row["critic/score/max"]) - float(row["critic/score/min"])
                row["zero_reward_variance_batch"] = float(abs(row["critic/score/range"]) < 1e-8)
            rows.append(row)
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame, group_cols: list[str], value_col: str = "value") -> pd.DataFrame:
    return df.groupby(group_cols, as_index=False)[value_col].agg(mean="mean", std="std", count="count")


def plot_step_curves(sampled: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["correct_pass", "citation_free_valid_pass", "citation_free_joint_pass", "citation_free_valid_given_correct"]
    titles = ["correct pass@1", "citation-free valid pass@1", "correct & cf-valid pass@1", "cf-valid given correct @1"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=False)
    for ax, metric, title in zip(axes.flat, metrics, titles, strict=True):
        df = sampled[(sampled.metric == metric) & (sampled.k == 1)]
        agg = mean_std(df, ["condition", "step"])
        for cond, sub in agg.groupby("condition"):
            sub = sub.sort_values("step")
            ax.plot(sub.step, sub["mean"], label=LABELS.get(cond, cond), color=COLORS.get(cond))
            ax.fill_between(sub.step, sub["mean"] - sub["std"].fillna(0), sub["mean"] + sub["std"].fillna(0), alpha=0.15, color=COLORS.get(cond))
        ax.axvspan(1, 5, color="#eeeeee", alpha=0.5, label="train depths" if metric == metrics[0] else None)
        ax.set_title(title)
        ax.set_xlabel("depth / required steps")
        ax.set_ylabel("rate")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "pass1_step_curves.pdf")
    fig.savefig(out_dir / "pass1_step_curves.png", dpi=180)
    plt.close(fig)


def plot_passk_slices(sampled: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["correct_pass", "citation_free_valid_pass", "citation_free_joint_pass"]
    steps = [10, 15, 20]
    fig, axes = plt.subplots(len(metrics), len(steps), figsize=(15, 9), sharex=True, sharey=True)
    for i, metric in enumerate(metrics):
        for j, step in enumerate(steps):
            ax = axes[i][j]
            df = sampled[(sampled.metric == metric) & (sampled.step == step)]
            agg = mean_std(df, ["condition", "k"])
            for cond, sub in agg.groupby("condition"):
                sub = sub.sort_values("k")
                ax.plot(sub.k, sub["mean"], marker="o", label=LABELS.get(cond, cond), color=COLORS.get(cond))
                ax.fill_between(sub.k, sub["mean"] - sub["std"].fillna(0), sub["mean"] + sub["std"].fillna(0), alpha=0.15, color=COLORS.get(cond))
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_title(f"{metric} @ depth {step}")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "passk_depth_slices.pdf")
    fig.savefig(out_dir / "passk_depth_slices.png", dpi=180)
    plt.close(fig)


def plot_training_logs(logs: pd.DataFrame, out_dir: Path) -> None:
    if logs.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    metrics = [
        ("train_aux/reward/correct/mean", "train rollout correctness"),
        ("train_aux/reward/citation_free_valid/mean", "train rollout citation-free validity"),
        ("train_aux/reward/citation_free_line_valid/mean", "train rollout cf line-valid fraction"),
        ("derived_main_reward_event", "main reward event density"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        df = logs.dropna(subset=[metric]) if metric in logs.columns else pd.DataFrame()
        if df.empty:
            continue
        agg = mean_std(df, ["condition", "step"], metric)
        for cond, sub in agg.groupby("condition"):
            sub = sub.sort_values("step")
            # Smooth just enough to make 500 per-step points readable.
            y = sub["mean"].rolling(15, min_periods=1, center=True).mean()
            ax.plot(sub.step, y, label=LABELS.get(cond, cond), color=COLORS.get(cond), linewidth=1.4)
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.13)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "train_rollout_reward_density.pdf")
    fig.savefig(out_dir / "train_rollout_reward_density.png", dpi=180)
    plt.close(fig)


def plot_classified(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if df.empty:
        return
    df = df.copy()
    if "probe_type" not in df.columns:
        df["probe_type"] = "normal"
    value_cols = [
        "correct",
        "citation_free_valid",
        "correct_gold_atom_derived_query",
        "correct_cfvalid_gold_atom_derived",
        "answer_derived_by_conclusion",
        "answer_is_shortcut",
        "proof_shorter_than_expected",
        "proof_line_ratio",
    ]
    present = [c for c in value_cols if c in df.columns]
    agg = df.groupby(["condition", "probe_type", "step"], as_index=False)[present].mean()
    agg.to_csv(out_dir / f"{prefix}_classified_rates.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = ["correct", "citation_free_valid", "correct_cfvalid_gold_atom_derived", "proof_shorter_than_expected"]
    titles = ["correct", "citation-free valid", "correct & cf-valid & derives gold atom", "shorter than expected proof"]
    for ax, metric, title in zip(axes.flat, metrics, titles, strict=True):
        if metric not in agg.columns:
            continue
        subdf = agg[agg.probe_type == "normal"]
        for cond, sub in subdf.groupby("condition"):
            sub = sub.sort_values("step")
            ax.plot(sub.step, sub[metric], marker="o", label=LABELS.get(cond, cond), color=COLORS.get(cond))
        ax.axvspan(1, 5, color="#eeeeee", alpha=0.5)
        ax.set_title(f"targeted normal: {title}")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / f"{prefix}_targeted_normal_rates.pdf")
    fig.savefig(out_dir / f"{prefix}_targeted_normal_rates.png", dpi=180)
    plt.close(fig)

    if "category" in df.columns:
        normal = df[df.probe_type == "normal"] if "probe_type" in df.columns else df
        stack = normal.groupby(["condition", "step", "category"]).size().reset_index(name="n")
        totals = stack.groupby(["condition", "step"])["n"].transform("sum")
        stack["rate"] = stack["n"] / totals
        stack.to_csv(out_dir / f"{prefix}_failure_category_rates.csv", index=False)
        for step in sorted(stack.step.unique()):
            if step not in {10, 15, 20}:
                continue
            pivot = stack[stack.step == step].pivot_table(index="condition", columns="category", values="rate", fill_value=0.0)
            pivot = pivot.reindex([c for c in LABELS if c in pivot.index])
            ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 5), colormap="tab20")
            ax.set_title(f"targeted normal failure categories at depth {step}")
            ax.set_ylabel("fraction")
            ax.set_xticklabels([LABELS.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()], rotation=20, ha="right")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_failure_stack_step{step}.pdf")
            plt.savefig(out_dir / f"{prefix}_failure_stack_step{step}.png", dpi=180)
            plt.close()


def write_examples(df: pd.DataFrame, out_dir: Path, max_per_category: int = 2) -> None:
    if df.empty or "generation" not in df.columns:
        return
    df = df.copy()
    if "probe_type" not in df.columns:
        df["probe_type"] = "normal"
    example_dir = out_dir / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    wanted_categories = [
        "correct_cfvalid_gold_atom_derived",
        "correct_cfvalid_no_gold_atom",
        "correct_cfinvalid_gold_atom_derived",
        "correct_cfinvalid_no_gold_atom",
        "shortcut_wrong",
        "wrong_cfvalid_gold_atom_derived",
        "wrong_cfvalid_no_gold_atom",
        "wrong_cfinvalid",
        "malformed",
    ]
    lines = ["# Qualitative Diagnostic Examples", ""]
    normal = df[df.get("probe_type", "normal") == "normal"] if "probe_type" in df.columns else df
    for cat in wanted_categories:
        sub = normal[normal.category == cat] if "category" in normal.columns else pd.DataFrame()
        if sub.empty:
            continue
        lines.append(f"## {cat}")
        for _, row in sub.sort_values(["step", "condition"]).head(max_per_category).iterrows():
            lines.append("")
            lines.append(f"- condition: `{row.get('condition')}`, step: `{row.get('step')}`, gold: `{row.get('gold_answer')}`, shortcut: `{row.get('shortcut_answer')}`, answer: `{row.get('answer_text')}`")
            lines.append(f"- cf_valid: `{row.get('citation_free_valid')}`, correct_answer_derived: `{row.get('correct_answer_derived')}`, proof lines: `{row.get('proof_line_count')}/{row.get('expected_proof_lines')}`")
            lines.append("\nPrompt excerpt:\n")
            prompt = str(row.get("prompt", ""))
            lines.append("```text")
            lines.append(prompt[:1600] + ("\n..." if len(prompt) > 1600 else ""))
            lines.append("```")
            lines.append("\nGeneration:\n")
            gen = str(row.get("generation", ""))
            lines.append("```text")
            lines.append(gen[:5000] + ("\n..." if len(gen) > 5000 else ""))
            lines.append("```")
    (example_dir / "qualitative_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_tables(sampled: pd.DataFrame, greedy: pd.DataFrame, logs: pd.DataFrame, classified_samples: pd.DataFrame, targeted: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if not sampled.empty:
        bands = {
            "train_1_5": sampled.step.between(1, 5),
            "ood_6_20": sampled.step.between(6, 20),
            "tail_15_20": sampled.step.between(15, 20),
        }
        rows = []
        for band, mask in bands.items():
            for metric in ["correct_pass", "citation_free_valid_pass", "citation_free_joint_pass", "citation_free_valid_given_correct", "citation_free_invalid_but_correct"]:
                for k in [1, 16, 64]:
                    sub = sampled[mask & (sampled.metric == metric) & (sampled.k == k)]
                    if sub.empty:
                        continue
                    per_seed = sub.groupby(["condition", "seed"], as_index=False).value.mean()
                    agg = per_seed.groupby("condition").value.agg(["mean", "std", "count"]).reset_index()
                    for _, r in agg.iterrows():
                        rows.append({"band": band, "metric": metric, "k": k, "condition": r["condition"], "mean": r["mean"], "std": r["std"], "count": r["count"]})
        band_df = pd.DataFrame(rows)
        band_df.to_csv(out_dir / "passk_band_summary.csv", index=False)
        summary["passk_band_summary_rows"] = len(band_df)
    if not greedy.empty:
        greedy.to_csv(out_dir / "greedy_metrics_long.csv", index=False)
    if not logs.empty:
        logs.to_csv(out_dir / "training_log_metrics_long.csv", index=False)
        final_logs = logs.sort_values("step").groupby(["condition", "seed"], as_index=False).tail(1)
        final_logs.to_csv(out_dir / "training_log_final_rows.csv", index=False)
    if not classified_samples.empty:
        classified_samples.to_csv(out_dir / "existing_greedy_samples_classified.csv", index=False)
    if not targeted.empty:
        targeted.to_csv(out_dir / "targeted_generations_classified.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hard_fsa_schema_easy500 validity/correctness diagnostics.")
    parser.add_argument("--passk-dir", default="/home/atuin/c107fa/c107fa12/synthetic-RLVL/passk_eval/hard_fsa_schema_easy500")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--targeted", nargs="*", default=[])
    parser.add_argument("--out-dir", default="analysis/hfsa_easy_validity_2026-05-14")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    sampled, greedy = load_passk(Path(args.passk_dir))
    classified_samples = classify_existing_samples(Path(args.passk_dir))
    logs = parse_training_logs(Path(args.log_dir))
    targeted = load_targeted([Path(p) for p in args.targeted])

    sampled.to_csv(table_dir / "sampled_passk_long.csv", index=False)
    greedy.to_csv(table_dir / "greedy_eval_long.csv", index=False)
    if not logs.empty:
        logs.to_csv(table_dir / "training_logs_long.csv", index=False)
    if not classified_samples.empty:
        classified_samples.to_csv(table_dir / "existing_samples_classified.csv", index=False)
    if not targeted.empty:
        targeted.to_csv(table_dir / "targeted_generations_raw.csv", index=False)

    plot_step_curves(sampled, fig_dir)
    plot_passk_slices(sampled, fig_dir)
    plot_training_logs(logs, fig_dir)
    plot_classified(classified_samples, fig_dir, "existing_samples")
    plot_classified(targeted, fig_dir, "targeted")
    write_examples(targeted if not targeted.empty else classified_samples, out_dir)
    summary = summarize_tables(sampled, greedy, logs, classified_samples, targeted, table_dir)

    print(json.dumps({
        "sampled_rows": len(sampled),
        "greedy_rows": len(greedy),
        "training_log_rows": len(logs),
        "classified_existing_sample_rows": len(classified_samples),
        "targeted_rows": len(targeted),
        "out_dir": str(out_dir),
        **summary,
    }, indent=2))


if __name__ == "__main__":
    main()
