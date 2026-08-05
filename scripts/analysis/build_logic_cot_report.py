from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


WORK_ROOT = Path(os.environ.get("WORK", ROOT))
PASSK_ROOT = WORK_ROOT / "synthetic-RLVL" / "passk_eval"
LM_EVAL_ROOT = WORK_ROOT / "synthetic-RLVL" / "lm_eval_results"
OUT_ROOT = ROOT / "analysis" / "logic_cot_report_2026-05-25"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
CORRECTED_BRANCHPROOF_ROOT = ROOT / "analysis" / "branchproof_unique_v2_20260711"
SELECTED_BRANCHPROOF_PASSK_ROOT = (
    Path(os.environ.get("HPCVAULT", WORK_ROOT))
    / "synthetic-RLVL"
    / "passk_eval"
    / "branchproof_unique_v2_report_20260713"
)
SELECTED_BRANCHPROOF_AUDIT_ROOT = (
    Path(os.environ.get("HPCVAULT", WORK_ROOT))
    / "synthetic-RLVL"
    / "analysis"
    / "branchproof_selected_followups_audits_20260723"
)
VERIFIER_BRANCHPROOF_PASSK_ROOT = (
    Path(os.environ.get("HPCVAULT", WORK_ROOT))
    / "synthetic-RLVL"
    / "passk_eval"
    / "branchproof_unique_v2_20260710"
)
VERIFIER_BRANCHPROOF_AUDIT_ROOT = (
    Path(os.environ.get("HPCVAULT", WORK_ROOT))
    / "synthetic-RLVL"
    / "analysis"
    / "branchproof_verifier_fullsamples_audits_20260805"
)
TOKENIZER_NAME = "allenai/Olmo-3-1025-7B"

MAIN_RE = re.compile(r"sft_hfsa_depth_scaling_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$")
MAIN_CKPT_RE = re.compile(
    r"sft_hfsa_depth_scaling_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_checkpoint-(\d+)_passk\.json$"
)
TINY_RE = re.compile(r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_seed(\d+)_passk\.json$")
TINY_CKPT_RE = re.compile(
    r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_seed(\d+)_checkpoint-(\d+)_passk\.json$"
)
TINY_100K_RE = re.compile(
    r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_100k_seed(\d+)_passk\.json$"
)
TINY_100K_CKPT_RE = re.compile(
    r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_100k_seed(\d+)_checkpoint-(\d+)_passk\.json$"
)
QWEN_RE = re.compile(
    r"sft_hfsa_modelablate_qwen2p5_7b_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
QWEN15_RE = re.compile(
    r"sft_hfsa_modelablate_qwen2p5_1p5b_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
GEMMA_RE = re.compile(
    r"sft_hfsa_modelablate_gemma3_4b_pt_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
OLMO32_PASSK_RE = re.compile(
    r"sft_hfsa_modelablate_olmo2_32b_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
MAIN_OOD_RE = re.compile(r"sft_hfsa_depth_scaling_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)$")
TINY_OOD_RE = re.compile(r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_seed(\d+)$")
TINY_100K_OOD_RE = re.compile(
    r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_100k_seed(\d+)$"
)
OLMO32_BARE_RE = re.compile(r"sft_hfsa_modelablate_olmo2_32b_(logic|nl_exact)_train1to20_10k_seed(\d+)$")
TOKBUDGET_RE = re.compile(
    r"sft_hfsa_same_target_tokens_(logic|nl_exact)_train1to25_(\d+)steps_seed(\d+)_passk\.json$"
)
SHORTCUT_RE = re.compile(
    r"sft_hfsa_shortcut_rate_(logic|nl_exact)_shortcut(0p3|0p5|0p8)_train1to25_10k_seed(\d+)_passk\.json$"
)
SHORTCUT_KIND_RE = re.compile(
    r"sft_hfsa_shortcut_(position|initial_marker)_(logic|nl_exact)_shortcut(0p5|0p8)_train1to25_10k_seed(\d+)_passk\.json$"
)
TRACE_CONTROL_RE = re.compile(
    r"sft_hfsa_ablate_(terse_nl|rule_annotated_nl|pseudocode|shuffled_logic|invalid_logic|shuffled_nl)_train1to25_10k_seed(\d+)_passk\.json$"
)
HYBRID_ORDER_RE = re.compile(
    r"sft_hfsa_hybrid_order_(think_formal|formal_think)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
CONDITIONED_RE = re.compile(
    r"sft_hfsa_conditioned_dual_train1to(\d+)_10k_seed(\d+)_(conditioned_logic|conditioned_nl)_passk\.json$"
)
CONDITIONED_50K_RE = re.compile(
    r"sft_hfsa_conditioned_dual_train1to(\d+)_50k_seed(\d+)_(conditioned_logic|conditioned_nl)_passk\.json$"
)
CONDITIONED_50K_CKPT_RE = re.compile(
    r"sft_hfsa_conditioned_dual_train1to25_50k_seed(\d+)_(conditioned_logic|conditioned_nl)_checkpoint-(\d+)_passk\.json$"
)
SYMBOL_PADDED_RE = re.compile(
    r"sft_hfsa_symbol_padded_(logic_symbol_padded)_train1to25_10k_seed(\d+)_passk\.json$"
)
WORDIFIED_RE = re.compile(
    r"sft_hfsa_wordified_(logic_wordified)_train1to25_10k_seed(\d+)_passk\.json$"
)
PAIRED_FULL_RE = re.compile(
    r"sft_paired_full_(official_igsm|maze_navigation|attribute_constraints_hard)_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
PAIRED_IGSM_SEMANTIC_RE = re.compile(
    r"sft_paired_igsm_semantic_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
PAIRED_MAZE_TYPED_RE = re.compile(
    r"sft_paired_maze_typed_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
PAIRED_HARD_ATTR_FRESH_RE = re.compile(
    r"sft_paired_full_attribute_constraints_hard_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
HFSA_BATCH_SIZE_RE = re.compile(
    r"sft_hfsa_batch_bsz(\d+)_(logic|nl_exact|conditioned_dual)_train1to(\d+)_10k_seed(\d+)_"
    r"(logic|nl_exact|conditioned_logic|conditioned_nl)_passk\.json$"
)

DEPTHS_FINAL = [1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
DEPTHS_TINY = [1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 40, 50]
OLMO_CKPT_DEPTH_BANDS = [("1-25", [1, 5, 10, 15, 20, 25]), ("30", [30]), ("40", [40]), ("50", [50])]
TINY_CKPT_DEPTH_BANDS = [
    ("1-10", [1, 2, 5, 10]),
    ("12-20", [12, 15, 18, 20]),
    ("25-30", [25, 30]),
    ("40-50", [40, 50]),
]
TEMPLATE_LABEL = {"logic": "Logic", "nl_exact": "NL exact"}
COLORS = {"logic": "#1f77b4", "nl_exact": "#d62728"}
SIZE_ORDER = {"50m": 50, "100m": 100, "200m": 200}
MODEL_LABELS = {
    "olmo7b": "OLMo-7B",
    "qwen7b": "Qwen-2.5-7B",
    "qwen1p5b": "Qwen-2.5-1.5B",
    "gemma4b": "Gemma-3-4B",
    "olmo32b_shortctx": "OLMo-2-32B shortctx",
}
MODEL_ORDER = {
    "OLMo-7B": 0,
    "Qwen-2.5-1.5B": 1,
    "Qwen-2.5-7B": 2,
    "Gemma-3-4B": 3,
    "OLMo-2-32B shortctx": 4,
}
MODEL_COLORS = {
    "OLMo-7B": "#1f77b4",
    "Qwen-2.5-1.5B": "#54a24b",
    "Qwen-2.5-7B": "#f58518",
    "Gemma-3-4B": "#b279a2",
    "OLMo-2-32B shortctx": "#4c78a8",
}


@dataclass(frozen=True)
class Record:
    source: str
    path: Path
    template: str
    train_max: int
    seed: int
    metrics: dict[str, float]
    size: str | None = None
    checkpoint: int | None = None


def joint_metric(template: str) -> str:
    return (
        "citation_free_joint_pass"
        if template in {"logic", "conditioned_logic", "logic_symbol_padded", "logic_wordified"}
        else "nl_logic_joint_pass"
    )


def valid_metric(template: str) -> str:
    return (
        "citation_free_valid_pass"
        if template in {"logic", "conditioned_logic", "logic_symbol_padded", "logic_wordified"}
        else "nl_logic_citation_free_valid_pass"
    )


def read_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(record: Record, suffix: str) -> float | None:
    value = record.metrics.get(f"synthetic_sampled/{suffix}")
    return float(value) if isinstance(value, (int, float)) else None


def band_metric(record: Record, band: str, name: str, k: int) -> float | None:
    return metric(record, f"band_{band}/{name}@{k}")


def step_metric(record: Record, depth: int, name: str, k: int) -> float | None:
    return metric(record, f"step_{depth}/{name}@{k}")


def depth_band_value(record: Record, depths: list[int], metric_name: str, k: int) -> float | None:
    if metric_name == "correct":
        values = [step_metric(record, depth, "correct_pass", k) for depth in depths]
    elif metric_name == "joint":
        values = [step_metric(record, depth, joint_metric(record.template), k) for depth in depths]
    else:
        raise ValueError(f"Unsupported depth-band metric: {metric_name}")
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not clean:
        return None
    return mean(clean)


def f3(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def avg_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return None, None, 0
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0, len(clean)


def load_records() -> tuple[
    list[Record],
    list[Record],
    list[Record],
    list[Record],
    list[Record],
    list[Record],
    list[Record],
]:
    main: list[Record] = []
    main_ckpt: list[Record] = []
    tiny: list[Record] = []
    tiny_ckpt: list[Record] = []
    tiny_100k: list[Record] = []
    tiny_100k_ckpt: list[Record] = []
    qwen: list[Record] = []

    for path in sorted((PASSK_ROOT / "hfsa_depth_scaling_sparse").glob("*_passk.json")):
        if match := MAIN_RE.match(path.name):
            payload = read_payload(path)
            main.append(
                Record("olmo7b", path, match.group(1), int(match.group(2)), int(match.group(3)), payload["metrics"])
            )
    for path in sorted((PASSK_ROOT / "hfsa_depth_scaling_intermediate_sparse").glob("*_passk.json")):
        if match := MAIN_CKPT_RE.match(path.name):
            payload = read_payload(path)
            main_ckpt.append(
                Record(
                    "olmo7b_ckpt",
                    path,
                    match.group(1),
                    int(match.group(2)),
                    int(match.group(3)),
                    payload["metrics"],
                    checkpoint=int(match.group(4)),
                )
            )
    for path in sorted((PASSK_ROOT / "hfsa_tiny_llama_pretrain_sparse").glob("*_passk.json")):
        if match := TINY_RE.match(path.name):
            payload = read_payload(path)
            tiny.append(
                Record(
                    "tiny_llama",
                    path,
                    match.group(2),
                    10,
                    int(match.group(3)),
                    payload["metrics"],
                    size=match.group(1),
                    checkpoint=20000,
                )
            )
    for path in sorted((PASSK_ROOT / "hfsa_tiny_llama_pretrain_intermediate_sparse").glob("*_passk.json")):
        if match := TINY_CKPT_RE.match(path.name):
            payload = read_payload(path)
            tiny_ckpt.append(
                Record(
                    "tiny_llama_ckpt",
                    path,
                    match.group(2),
                    10,
                    int(match.group(3)),
                    payload["metrics"],
                    size=match.group(1),
                    checkpoint=int(match.group(4)),
                )
            )
    for path in sorted((PASSK_ROOT / "hfsa_tiny_llama_pretrain_100k_sparse").glob("*_passk.json")):
        if match := TINY_100K_RE.match(path.name):
            payload = read_payload(path)
            tiny_100k.append(
                Record(
                    "tiny_llama_100k",
                    path,
                    match.group(2),
                    10,
                    int(match.group(3)),
                    payload["metrics"],
                    size=match.group(1),
                    checkpoint=100000,
                )
            )
    for path in sorted((PASSK_ROOT / "hfsa_tiny_llama_pretrain_100k_intermediate_sparse").glob("*_passk.json")):
        if match := TINY_100K_CKPT_RE.match(path.name):
            payload = read_payload(path)
            tiny_100k_ckpt.append(
                Record(
                    "tiny_llama_100k_ckpt",
                    path,
                    match.group(2),
                    10,
                    int(match.group(3)),
                    payload["metrics"],
                    size=match.group(1),
                    checkpoint=int(match.group(4)),
                )
            )
    for path in sorted((PASSK_ROOT / "hfsa_model_ablation_qwen2p5_7b_sparse").glob("*_passk.json")):
        if match := QWEN_RE.match(path.name):
            payload = read_payload(path)
            qwen.append(
                Record("qwen7b", path, match.group(1), int(match.group(2)), int(match.group(3)), payload["metrics"])
            )
    return main, main_ckpt, tiny, tiny_ckpt, tiny_100k, tiny_100k_ckpt, qwen


def load_extra_architecture_records() -> list[Record]:
    records: list[Record] = []
    specs = [
        (PASSK_ROOT / "hfsa_model_ablation_qwen2p5_1p5b_sparse", QWEN15_RE, "qwen1p5b"),
        (PASSK_ROOT / "hfsa_model_ablation_gemma3_4b_pt_sparse", GEMMA_RE, "gemma4b"),
        (PASSK_ROOT / "hfsa_model_ablation_olmo2_32b_shortctx_sparse", OLMO32_PASSK_RE, "olmo32b_shortctx"),
    ]
    for root, regex, source in specs:
        for path in sorted(root.glob("*_passk.json")):
            if match := regex.match(path.name):
                payload = read_payload(path)
                records.append(
                    Record(source, path, match.group(1), int(match.group(2)), int(match.group(3)), payload["metrics"])
                )
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(records: list[Record], ks: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, int, str | None], list[Record]] = defaultdict(list)
    for record in records:
        groups[(record.template, record.train_max, record.size)].append(record)
    for (template, train_max, size), items in sorted(
        groups.items(), key=lambda x: (x[0][2] or "", x[0][0], x[0][1])
    ):
        row: dict[str, object] = {
            "size": size or "",
            "template": template,
            "train_max": train_max,
            "n": len(items),
        }
        for k in ks:
            for band in ["train", "ood", "hard_tail"]:
                for name, metric_name in [
                    ("correct", "correct_pass"),
                    ("valid", valid_metric(template)),
                    ("joint", joint_metric(template)),
                ]:
                    avg, std, _ = avg_std([band_metric(item, band, metric_name, k) for item in items])
                    row[f"{band}_{name}@{k}"] = avg
                    row[f"{band}_{name}@{k}_std"] = std
            avg, std, _ = avg_std([step_metric(item, 50, "correct_pass", k) for item in items])
            row[f"depth50_correct@{k}"] = avg
            row[f"depth50_correct@{k}_std"] = std
            avg, std, _ = avg_std([step_metric(item, 50, joint_metric(template), k) for item in items])
            row[f"depth50_joint@{k}"] = avg
            row[f"depth50_joint@{k}_std"] = std
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(TABLE_DIR / "group_summary_all.csv", index=False, lineterminator="\n")
    return df


def summarize_architecture(records: list[Record]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str, int], list[Record]] = defaultdict(list)
    for record in records:
        groups[(record.source, record.template, record.train_max)].append(record)
    for (source, template, train_max), items in sorted(
        groups.items(), key=lambda x: (MODEL_ORDER.get(MODEL_LABELS.get(x[0][0], x[0][0]), 99), x[0][1], x[0][2])
    ):
        row: dict[str, object] = {
            "model": MODEL_LABELS.get(source, source),
            "template": template,
            "train_max": train_max,
            "n": len({item.seed for item in items}),
        }
        for col, values in {
            "ood_correct@16": [band_metric(item, "ood", "correct_pass", 16) for item in items],
            "ood_joint@16": [band_metric(item, "ood", joint_metric(template), 16) for item in items],
            "depth30_50_correct@16": [
                depth_band_value(item, [30, 35, 40, 45, 50], "correct", 16) for item in items
            ],
            "depth30_50_joint@16": [
                depth_band_value(item, [30, 35, 40, 45, 50], "joint", 16) for item in items
            ],
            "depth50_correct@16": [step_metric(item, 50, "correct_pass", 16) for item in items],
            "depth50_joint@16": [step_metric(item, 50, joint_metric(template), 16) for item in items],
            "shortctx_correct@16": [
                depth_band_value(item, [1, 2, 5, 10, 12, 15], "correct", 16) for item in items
            ],
            "shortctx_joint@16": [
                depth_band_value(item, [1, 2, 5, 10, 12, 15], "joint", 16) for item in items
            ],
        }.items():
            avg, std, _ = avg_std(values)
            row[col] = avg
            row[f"{col}_std"] = std
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        write_csv(TABLE_DIR / "architecture_ablation_summary.csv", df.to_dict("records"))
    return df


def depth_dataframe(records: list[Record], depths: list[int], k: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        for depth in depths:
            rows.append(
                {
                    "source": record.source,
                    "size": record.size or "",
                    "template": record.template,
                    "train_max": record.train_max,
                    "seed": record.seed,
                    "checkpoint": record.checkpoint or 0,
                    "depth": depth,
                    "correct": step_metric(record, depth, "correct_pass", k),
                    "valid": step_metric(record, depth, valid_metric(record.template), k),
                    "joint": step_metric(record, depth, joint_metric(record.template), k),
                }
            )
    return pd.DataFrame(rows)


def style_axes(ax, ylabel: str, title: str | None = None) -> None:
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=180)
    plt.close(fig)


def plot_main_trainmax(summary: pd.DataFrame) -> None:
    main = summary[(summary["size"] == "") & (summary["n"] >= 1)].copy()
    if main.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    panels = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("hard_tail_correct@16", "hard-tail correct@16"),
        ("hard_tail_joint@16", "hard-tail joint@16"),
    ]
    for ax, (col, label) in zip(axes.ravel(), panels, strict=True):
        for template in ["logic", "nl_exact"]:
            sub = main[main["template"] == template].sort_values("train_max")
            if sub.empty or col not in sub:
                continue
            ax.plot(sub["train_max"], sub[col], marker="o", color=COLORS[template], label=TEMPLATE_LABEL[template])
            std_col = f"{col}_std"
            if std_col in sub:
                ax.fill_between(
                    sub["train_max"],
                    (sub[col] - sub[std_col]).clip(0, 1),
                    (sub[col] + sub[std_col]).clip(0, 1),
                    color=COLORS[template],
                    alpha=0.12,
                    linewidth=0,
                )
        style_axes(ax, label)
        ax.set_xlabel("max train depth")
    axes[0, 0].legend(frameon=False, loc="lower right")
    save(fig, "olmo7b_final_by_train_depth")


def plot_depth_grid(depth_df: pd.DataFrame, name: str, metric_name: str, k: int) -> None:
    if depth_df.empty:
        return
    grouped = depth_df.groupby(["template", "train_max", "depth"], as_index=False)[metric_name].mean()
    train_maxes = sorted(grouped["train_max"].unique())
    fig, axes = plt.subplots(1, len(train_maxes), figsize=(3.1 * len(train_maxes), 3.4), sharey=True)
    if len(train_maxes) == 1:
        axes = [axes]
    for ax, train_max in zip(axes, train_maxes, strict=False):
        for template in ["logic", "nl_exact"]:
            sub = grouped[(grouped["train_max"] == train_max) & (grouped["template"] == template)].sort_values("depth")
            if sub.empty:
                continue
            ax.plot(sub["depth"], sub[metric_name], marker="o", markersize=3, color=COLORS[template], label=TEMPLATE_LABEL[template])
        ax.axvline(train_max, color="black", linestyle=":", linewidth=1)
        style_axes(ax, f"{metric_name}@{k}", f"train 1..{train_max}")
        ax.set_xlabel("eval depth")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, name)


def load_symbol_padded_records() -> list[Record]:
    records: list[Record] = []
    root = PASSK_ROOT / "hfsa_logic_symbol_padded_20260528"
    if not root.exists():
        return records
    for path in sorted(root.glob("*_passk.json")):
        match = SYMBOL_PADDED_RE.match(path.name)
        if not match:
            continue
        payload = read_payload(path)
        records.append(
            Record(
                "logic_symbol_padded",
                path,
                match.group(1),
                25,
                int(match.group(2)),
                payload["metrics"],
            )
        )
    return records


def load_wordified_records() -> list[Record]:
    records: list[Record] = []
    root = PASSK_ROOT / "hfsa_logic_wordified_20260529"
    if not root.exists():
        return records
    for path in sorted(root.glob("*_passk.json")):
        match = WORDIFIED_RE.match(path.name)
        if not match:
            continue
        payload = read_payload(path)
        records.append(
            Record(
                "logic_wordified",
                path,
                match.group(1),
                25,
                int(match.group(2)),
                payload["metrics"],
            )
        )
    return records


def symbol_padded_depth_dataframe(
    main_records: list[Record],
    symbol_records: list[Record],
    wordified_records: list[Record] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    conditions: list[tuple[str, Record]] = []
    for record in main_records:
        if record.train_max == 25 and record.template in {"logic", "nl_exact"}:
            label = "compact logic" if record.template == "logic" else "NL exact"
            conditions.append((label, record))
    for record in symbol_records:
        conditions.append(("symbol-padded logic", record))
    for record in wordified_records or []:
        conditions.append(("wordified logic", record))

    for condition, record in conditions:
        for depth in DEPTHS_FINAL:
            rows.append(
                {
                    "condition": condition,
                    "template": record.template,
                    "seed": record.seed,
                    "train_max": record.train_max,
                    "depth": depth,
                    "correct@16": step_metric(record, depth, "correct_pass", 16),
                    "joint@16": step_metric(record, depth, joint_metric(record.template), 16),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        write_csv(TABLE_DIR / "logic_symbol_padded_depth_curve_vs_main_train25.csv", df.to_dict("records"))
        write_csv(TABLE_DIR / "logic_length_control_depth_curve_vs_main_train25.csv", df.to_dict("records"))
    return df


def plot_symbol_padded_depth_comparison(
    main_records: list[Record],
    symbol_records: list[Record],
    wordified_records: list[Record] | None = None,
) -> None:
    df = symbol_padded_depth_dataframe(main_records, symbol_records, wordified_records)
    if df.empty:
        return
    grouped = df.groupby(["condition", "depth"], as_index=False)[["correct@16", "joint@16"]].mean()
    styles = {
        "compact logic": ("#1f77b4", "-"),
        "symbol-padded logic": ("#54a24b", "--"),
        "wordified logic": ("#9467bd", "-."),
        "NL exact": ("#d62728", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for ax, (metric_name, ylabel) in zip(
        axes,
        [("correct@16", "correct@16"), ("joint@16", "joint correct+valid@16")],
        strict=True,
    ):
        for condition, (color, linestyle) in styles.items():
            sub = grouped[grouped["condition"] == condition].sort_values("depth")
            if sub.empty:
                continue
            ax.plot(
                sub["depth"],
                sub[metric_name],
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=color,
                linestyle=linestyle,
                label=condition,
            )
        ax.axvline(25, color="black", linestyle=":", linewidth=1)
        style_axes(ax, ylabel)
        ax.set_xlabel("eval depth")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_symbol_padded_depth_curve_train1to25.pdf")
    fig.savefig(FIG_DIR / "ablation_symbol_padded_depth_curve_train1to25.png", dpi=180)
    save(fig, "ablation_logic_length_control_depth_curve_train1to25")


def plot_main_checkpoint_curves(records: list[Record]) -> None:
    if not records:
        return
    rows = []
    for record in records:
        for band in ["train", "ood", "hard_tail"]:
            rows.append(
                {
                    "template": record.template,
                    "train_max": record.train_max,
                    "checkpoint": record.checkpoint,
                    "band": band,
                    "correct": band_metric(record, band, "correct_pass", 16),
                    "joint": band_metric(record, band, joint_metric(record.template), 16),
                }
            )
    df = pd.DataFrame(rows)
    for train_max in sorted(df["train_max"].unique()):
        sub_train = df[df["train_max"] == train_max]
        for metric_name in ["correct", "joint"]:
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=True, sharey=True)
            for ax, band in zip(axes, ["train", "ood", "hard_tail"], strict=True):
                sub0 = sub_train[sub_train["band"] == band]
                for template in ["logic", "nl_exact"]:
                    sub = sub0[sub0["template"] == template].sort_values("checkpoint")
                    if sub.empty:
                        continue
                    ax.plot(
                        sub["checkpoint"],
                        sub[metric_name],
                        color=COLORS[template],
                        marker="o",
                        markersize=3,
                        label=TEMPLATE_LABEL[template],
                    )
                style_axes(ax, f"{metric_name}@16", band)
                ax.set_xlabel("optimizer step")
            axes[0].legend(fontsize=8, frameon=False)
            save(fig, f"olmo7b_checkpoint_train1to{train_max}_{metric_name}16")


def plot_main_checkpoint_depth_bands(records: list[Record]) -> None:
    sub_records = [record for record in records if record.train_max == 25 and record.seed == 3407]
    if not sub_records:
        return
    for metric_name in ["correct", "joint"]:
        fig, axes = plt.subplots(1, len(OLMO_CKPT_DEPTH_BANDS), figsize=(12.8, 3.4), sharex=True, sharey=True)
        for ax, (label, depths) in zip(axes, OLMO_CKPT_DEPTH_BANDS, strict=True):
            for template in ["logic", "nl_exact"]:
                rows = []
                for record in sub_records:
                    if record.template != template or record.checkpoint is None:
                        continue
                    rows.append(
                        {
                            "checkpoint": record.checkpoint,
                            "value": depth_band_value(record, depths, metric_name, 16),
                        }
                    )
                df = pd.DataFrame(rows).dropna()
                if df.empty:
                    continue
                df = df.sort_values("checkpoint")
                ax.plot(
                    df["checkpoint"],
                    df["value"],
                    marker="o",
                    markersize=3,
                    color=COLORS[template],
                    label=TEMPLATE_LABEL[template],
                )
            style_axes(ax, f"{metric_name}@16", f"eval depth {label}")
            ax.set_xlabel("optimizer step")
        axes[0].legend(fontsize=8, frameon=False)
        save(fig, f"olmo7b_checkpoint_train1to25_depthbands_{metric_name}16")


def plot_tiny_final(tiny: list[Record]) -> None:
    if not tiny:
        return
    df = summarize_group(tiny, (8,))
    df = df[df["size"] != ""].copy()
    band_labels = ["train", "OOD", "hard", "d50"]
    for size in ["50m", "100m", "200m"]:
        sub_size = df[df["size"] == size]
        if sub_size.empty:
            continue
        for metric_name, label in [("correct", "correct@8"), ("joint", "joint@8")]:
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for template in ["logic", "nl_exact"]:
                row = sub_size[sub_size["template"] == template]
                if row.empty:
                    continue
                row0 = row.iloc[0]
                values = [
                    row0.get(f"train_{metric_name}@8"),
                    row0.get(f"ood_{metric_name}@8"),
                    row0.get(f"hard_tail_{metric_name}@8"),
                    row0.get(f"depth50_{metric_name}@8"),
                ]
                ax.plot(band_labels, values, marker="o", color=COLORS[template], label=TEMPLATE_LABEL[template])
            style_axes(ax, label, size)
            ax.set_xlabel("eval band")
            ax.legend(frameon=False)
            save(fig, f"tiny_llama_{size}_bands_{metric_name}_k8")

    depth_df = depth_dataframe(tiny, DEPTHS_TINY, 8)
    depth_plot_df = depth_df.groupby(["size", "template", "depth"], as_index=False)[["correct", "joint"]].mean()
    for size in ["50m", "100m", "200m"]:
        sub = depth_plot_df[depth_plot_df["size"] == size]
        if sub.empty:
            continue
        for metric_name in ["correct", "joint"]:
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for template in ["logic", "nl_exact"]:
                s = sub[sub["template"] == template].sort_values("depth")
                ax.plot(s["depth"], s[metric_name], marker="o", color=COLORS[template], label=TEMPLATE_LABEL[template])
            ax.axvline(10, color="black", linestyle=":", linewidth=1)
            style_axes(ax, f"{metric_name}@8", f"{size}")
            ax.set_xlabel("eval depth")
            ax.legend(frameon=False)
            save(fig, f"tiny_llama_{size}_depth_{metric_name}_k8")


def plot_tiny_checkpoint_curves(tiny_final: list[Record], tiny_ckpt: list[Record]) -> None:
    records = tiny_ckpt + tiny_final
    if not tiny_ckpt:
        return
    rows = []
    for record in records:
        for band in ["train", "ood", "hard_tail"]:
            rows.append(
                {
                    "size": record.size,
                    "template": record.template,
                    "checkpoint": record.checkpoint,
                    "band": band,
                    "correct": band_metric(record, band, "correct_pass", 8),
                    "joint": band_metric(record, band, joint_metric(record.template), 8),
                }
            )
    df = pd.DataFrame(rows)
    plot_df = df.groupby(["size", "template", "checkpoint", "band"], as_index=False)[["correct", "joint"]].mean()
    for size in ["50m", "100m", "200m"]:
        sub_size = plot_df[plot_df["size"] == size]
        if sub_size.empty:
            continue
        for metric_name in ["correct", "joint"]:
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
            for ax, band in zip(axes, ["train", "ood", "hard_tail"], strict=True):
                sub0 = sub_size[sub_size["band"] == band]
                for template in ["logic", "nl_exact"]:
                    sub = sub0[sub0["template"] == template].sort_values("checkpoint")
                    if sub.empty:
                        continue
                    ax.plot(
                        sub["checkpoint"],
                        sub[metric_name],
                        marker="o",
                        color=COLORS[template],
                        label=TEMPLATE_LABEL[template] if band == "train" else None,
                    )
                style_axes(ax, f"{metric_name}@8", band)
                ax.set_xlabel("pretraining step")
            axes[0].legend(fontsize=8, frameon=False)
            save(fig, f"tiny_llama_{size}_checkpoint_{metric_name}_k8")


def plot_tiny_checkpoint_depth_bands(
    tiny_final: list[Record],
    tiny_ckpt: list[Record],
    *,
    prefix: str,
) -> None:
    records = tiny_ckpt + tiny_final
    if not tiny_ckpt:
        return
    for size in ["50m", "100m", "200m"]:
        sub_records = [record for record in records if record.size == size]
        if not sub_records:
            continue
        for metric_name in ["correct", "joint"]:
            fig, axes = plt.subplots(1, len(TINY_CKPT_DEPTH_BANDS), figsize=(12.8, 3.4), sharex=True, sharey=True)
            for ax, (label, depths) in zip(axes, TINY_CKPT_DEPTH_BANDS, strict=True):
                for template in ["logic", "nl_exact"]:
                    rows = []
                    for record in sub_records:
                        if record.template != template or record.checkpoint is None:
                            continue
                        rows.append(
                            {
                                "checkpoint": record.checkpoint,
                                "value": depth_band_value(record, depths, metric_name, 8),
                            }
                        )
                    df = pd.DataFrame(rows).dropna()
                    if df.empty:
                        continue
                    df = df.groupby("checkpoint", as_index=False)["value"].mean().sort_values("checkpoint")
                    ax.plot(
                        df["checkpoint"],
                        df["value"],
                        marker="o",
                        markersize=3,
                        color=COLORS[template],
                        label=TEMPLATE_LABEL[template],
                    )
                style_axes(ax, f"{metric_name}@8", f"eval depth {label}")
                ax.set_xlabel("pretraining step")
            axes[0].legend(fontsize=8, frameon=False)
            save(fig, f"{prefix}_{size}_checkpoint_depthbands_{metric_name}_k8")


def plot_qwen_partial(qwen: list[Record]) -> None:
    if not qwen:
        return
    df = summarize_group(qwen, (16,))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    for ax, col in zip(axes, ["ood_correct@16", "ood_joint@16"], strict=True):
        for template in ["logic", "nl_exact"]:
            sub = df[df["template"] == template].sort_values("train_max")
            if sub.empty:
                continue
            ax.plot(sub["train_max"], sub[col], marker="o", color=COLORS[template], label=TEMPLATE_LABEL[template])
        style_axes(ax, col)
        ax.set_xlabel("max train depth")
    axes[0].legend(frameon=False)
    save(fig, "qwen7b_partial_ood_correct_joint")


def plot_architecture_comparison(architecture_summary: pd.DataFrame) -> None:
    if architecture_summary.empty:
        return
    plot_specs = [
        ("ood_correct@16", "OOD correct@16", "architecture_ood_correct16_by_train_depth"),
        ("depth30_50_correct@16", "eval-depth 30-50 correct@16", "architecture_depth30_50_correct16_by_train_depth"),
    ]
    for col, ylabel, name in plot_specs:
        if col not in architecture_summary:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
        for ax, template in zip(axes, ["logic", "nl_exact"], strict=True):
            sub_template = architecture_summary[architecture_summary["template"] == template]
            for model in sorted(sub_template["model"].unique(), key=lambda m: MODEL_ORDER.get(m, 99)):
                sub = sub_template[sub_template["model"] == model].dropna(subset=[col]).sort_values("train_max")
                if sub.empty:
                    continue
                ax.plot(
                    sub["train_max"],
                    sub[col],
                    marker="o",
                    color=MODEL_COLORS.get(model, "#666666"),
                    label=model,
                )
            style_axes(ax, ylabel, TEMPLATE_LABEL.get(template, template))
            ax.set_xlabel("max train depth")
        axes[0].legend(frameon=False, fontsize=8)
        save(fig, name)


def plot_token_budget_comparison(main_summary: pd.DataFrame, token_budget_summary: pd.DataFrame) -> None:
    if main_summary.empty or token_budget_summary.empty:
        return
    rows: list[dict[str, object]] = []
    baseline = main_summary[(main_summary["size"] == "") & (main_summary["train_max"] == 25)]
    for _, row in baseline.iterrows():
        rows.append(
            {
                "condition": f"main {row['template']} 10k",
                "ood_correct@16": row.get("ood_correct@16"),
                "ood_joint@16": row.get("ood_joint@16"),
                "depth50_correct@16": row.get("depth50_correct@16"),
                "depth50_joint@16": row.get("depth50_joint@16"),
            }
        )
    for _, row in token_budget_summary.iterrows():
        rows.append(
            {
                "condition": f"tok {row['template']} {int(row['steps'])} steps",
                "ood_correct@16": row.get("ood_correct@16"),
                "ood_joint@16": row.get("ood_joint@16"),
                "depth50_correct@16": row.get("depth50_correct@16"),
                "depth50_joint@16": row.get("depth50_joint@16"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharey=True)
    for ax, (col, label) in zip(axes.ravel(), metrics, strict=True):
        ax.barh(df["condition"], df[col], color=["#1f77b4", "#d62728", "#4c78a8", "#e45756"])
        style_axes(ax, label)
        ax.set_xlabel(label)
    save(fig, "ablation_same_target_token_budget_vs_main")


def plot_shortcut_comparison(main_summary: pd.DataFrame, shortcut_summary: pd.DataFrame) -> None:
    if main_summary.empty:
        return
    rows: list[dict[str, object]] = []
    baseline = main_summary[(main_summary["size"] == "") & (main_summary["train_max"] == 25)]
    for _, row in baseline.iterrows():
        rows.append(
            {
                "shortcut_rate": 0.0,
                "template": row["template"],
                "ood_correct@16": row.get("ood_correct@16"),
                "ood_joint@16": row.get("ood_joint@16"),
                "depth50_correct@16": row.get("depth50_correct@16"),
                "depth50_joint@16": row.get("depth50_joint@16"),
            }
        )
    if not shortcut_summary.empty:
        for _, row in shortcut_summary.iterrows():
            rate = str(row["shortcut_rate"]).replace("0p", "0.")
            rows.append(
                {
                    "shortcut_rate": float(rate),
                    "template": row["template"],
                    "ood_correct@16": row.get("ood_correct@16"),
                    "ood_joint@16": row.get("ood_joint@16"),
                    "depth50_correct@16": row.get("depth50_correct@16"),
                    "depth50_joint@16": row.get("depth50_joint@16"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True, sharey=True)
    for ax, (col, label) in zip(axes.ravel(), metrics, strict=True):
        for template in ["logic", "nl_exact"]:
            sub = df[df["template"] == template].sort_values("shortcut_rate")
            if sub.empty:
                continue
            ax.plot(
                sub["shortcut_rate"],
                sub[col],
                marker="o",
                color=COLORS[template],
                label=TEMPLATE_LABEL[template],
            )
        style_axes(ax, label)
        ax.set_xlabel("train shortcut rate")
    axes[0, 0].legend(frameon=False)
    save(fig, "ablation_shortcut_rate_vs_main")


def plot_shortcut_kind_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    df = summary.copy()
    df["shortcut_rate_value"] = df["shortcut_rate"].astype(str).str.replace("0p", "0.", regex=False).astype(float)
    df["condition"] = (
        df["shortcut_kind"].astype(str).str.replace("_", " ", regex=False)
        + "\nrate "
        + df["shortcut_rate_value"].map(lambda value: f"{value:.1f}")
    )
    condition_order = (
        df[["shortcut_kind", "shortcut_rate_value", "condition"]]
        .drop_duplicates()
        .sort_values(["shortcut_kind", "shortcut_rate_value"])
    )
    conditions = list(condition_order["condition"])
    x = list(range(len(conditions)))
    width = 0.36
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharey=True)
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        for offset, template in [(-width / 2, "logic"), (width / 2, "nl_exact")]:
            values = []
            for condition in conditions:
                match = df[(df["condition"] == condition) & (df["template"] == template)]
                value = match.iloc[0][col] if not match.empty else None
                values.append(float(value) if isinstance(value, Number) and not pd.isna(value) else 0.0)
            ax.bar([pos + offset for pos in x], values, width=width, color=COLORS[template], label=TEMPLATE_LABEL[template])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=8)
        style_axes(ax, title)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "ablation_shortcut_kind_summary")


def plot_shortcut_kind_lines(main_summary: pd.DataFrame, summary: pd.DataFrame) -> None:
    if main_summary.empty or summary.empty:
        return
    rows: list[dict[str, object]] = []
    baseline = main_summary[(main_summary["size"] == "") & (main_summary["train_max"] == 25)]
    kinds = sorted(str(value) for value in summary["shortcut_kind"].dropna().unique())
    for kind in kinds:
        for _, row in baseline.iterrows():
            if row["template"] not in {"logic", "nl_exact"}:
                continue
            rows.append(
                {
                    "shortcut_kind": kind,
                    "shortcut_rate_value": 0.0,
                    "template": row["template"],
                    "ood_correct@16": row.get("ood_correct@16"),
                    "ood_joint@16": row.get("ood_joint@16"),
                    "depth50_correct@16": row.get("depth50_correct@16"),
                    "depth50_joint@16": row.get("depth50_joint@16"),
                }
            )
    for _, row in summary.iterrows():
        rows.append(
            {
                "shortcut_kind": row["shortcut_kind"],
                "shortcut_rate_value": float(str(row["shortcut_rate"]).replace("0p", "0.")),
                "template": row["template"],
                "ood_correct@16": row.get("ood_correct@16"),
                "ood_joint@16": row.get("ood_joint@16"),
                "depth50_correct@16": row.get("depth50_correct@16"),
                "depth50_joint@16": row.get("depth50_joint@16"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    write_csv(TABLE_DIR / "shortcut_kind_ablation_vs_main.csv", df.to_dict("records"))
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    fig, axes = plt.subplots(len(kinds), len(metrics), figsize=(13.5, 3.8 * len(kinds)), sharex=True, sharey=True)
    if len(kinds) == 1:
        axes = axes.reshape(1, -1)
    for row_idx, kind in enumerate(kinds):
        for col_idx, (metric, label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for template in ["logic", "nl_exact"]:
                sub = df[(df["shortcut_kind"] == kind) & (df["template"] == template)].sort_values(
                    "shortcut_rate_value"
                )
                if sub.empty:
                    continue
                ax.plot(
                    sub["shortcut_rate_value"],
                    sub[metric],
                    marker="o",
                    color=COLORS[template],
                    label=TEMPLATE_LABEL[template],
                )
            style_axes(ax, f"{kind.replace('_', ' ')}\n{label}")
            ax.set_xlabel("train shortcut rate")
            ax.set_xticks([0.0, 0.5, 0.8])
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "ablation_shortcut_kind_rate_lines_vs_main")


def build_shortcut_comparison_table(main_summary: pd.DataFrame, shortcut_summary: pd.DataFrame) -> pd.DataFrame:
    if main_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    metric_cols = ["ood_correct@16", "ood_joint@16", "depth50_correct@16", "depth50_joint@16"]
    baseline = main_summary[(main_summary["size"] == "") & (main_summary["train_max"] == 25)]
    for _, row in baseline.sort_values("template").iterrows():
        item: dict[str, object] = {"template": row["template"], "shortcut_rate": 0.0, "n": row.get("n")}
        for col in metric_cols:
            item[col] = row.get(col)
        rows.append(item)
    if not shortcut_summary.empty:
        for _, row in shortcut_summary.sort_values(["template", "shortcut_rate"]).iterrows():
            item = {
                "template": row["template"],
                "shortcut_rate": float(str(row["shortcut_rate"]).replace("0p", "0.")),
                "n": row.get("n"),
            }
            for col in metric_cols:
                item[col] = row.get(col)
            rows.append(item)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["template", "shortcut_rate"])
        write_csv(TABLE_DIR / "shortcut_rate_ablation_vs_main.csv", df.to_dict("records"))
    return df


def build_conditioned_comparison_table(main_summary: pd.DataFrame, conditioned_summary: pd.DataFrame) -> pd.DataFrame:
    if main_summary.empty and conditioned_summary.empty:
        return pd.DataFrame()
    metric_cols = ["ood_correct@16", "ood_joint@16", "depth50_correct@16", "depth50_joint@16"]
    rows: list[dict[str, object]] = []
    main_rows = main_summary[
        (main_summary.get("size", "") == "")
        & (main_summary.get("template").isin(["logic", "nl_exact"]))
        & (main_summary.get("train_max").isin([5, 10, 15, 20, 25]))
    ]
    for _, row in main_rows.iterrows():
        template = str(row["template"])
        item: dict[str, object] = {
            "train_max": int(row["train_max"]),
            "condition": "main logic" if template == "logic" else "main NL exact",
            "modality": "logic" if template == "logic" else "nl",
            "n": row.get("n"),
        }
        for col in metric_cols:
            item[col] = row.get(col)
        rows.append(item)
    if not conditioned_summary.empty:
        for _, row in conditioned_summary.iterrows():
            eval_template = str(row["eval_template"])
            item = {
                "train_max": int(row["train_max"]),
                "condition": "conditioned logic" if eval_template == "conditioned_logic" else "conditioned NL",
                "modality": "logic" if eval_template == "conditioned_logic" else "nl",
                "n": row.get("n"),
            }
            for col in metric_cols:
                item[col] = row.get(col)
            rows.append(item)
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"main logic": 0, "conditioned logic": 1, "main NL exact": 2, "conditioned NL": 3}
        df["condition_order"] = df["condition"].map(order)
        df = df.sort_values(["train_max", "condition_order"]).drop(columns=["condition_order"])
        write_csv(TABLE_DIR / "conditioned_dual_vs_main_by_train_depth.csv", df.to_dict("records"))
    return df


def plot_conditioned_comparison(conditioned_comparison: pd.DataFrame) -> None:
    if conditioned_comparison.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    styles = {
        "main logic": ("#1f77b4", "-"),
        "conditioned logic": ("#1f77b4", "--"),
        "main NL exact": ("#d62728", "-"),
        "conditioned NL": ("#d62728", "--"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), sharex=True, sharey=True)
    for ax, (col, label) in zip(axes.ravel(), metrics, strict=True):
        for condition, (color, linestyle) in styles.items():
            sub = conditioned_comparison[conditioned_comparison["condition"] == condition].sort_values("train_max")
            if sub.empty:
                continue
            ax.plot(
                sub["train_max"],
                sub[col],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=condition,
            )
        style_axes(ax, label)
        ax.set_xlabel("max train depth")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "ablation_conditioned_dual_vs_main_by_train_depth")


def load_conditioned_50k_checkpoint_summary() -> pd.DataFrame:
    root = PASSK_ROOT / "hfsa_conditioned_dual_50k_intermediate_20260529"
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*_passk.json")):
        match = CONDITIONED_50K_CKPT_RE.match(path.name)
        if not match:
            continue
        seed = int(match.group(1))
        eval_template = match.group(2)
        checkpoint = int(match.group(3))
        payload = read_payload(path)
        joint = _joint_key(eval_template)
        rows.append(
            {
                "seed": seed,
                "eval_template": eval_template,
                "checkpoint": checkpoint,
                "ood_correct@16": payload["metrics"].get("synthetic_sampled/band_ood/correct_pass@16"),
                "ood_joint@16": payload["metrics"].get(f"synthetic_sampled/band_ood/{joint}@16"),
                "depth50_correct@16": payload["metrics"].get("synthetic_sampled/step_50/correct_pass@16"),
                "depth50_joint@16": payload["metrics"].get(f"synthetic_sampled/step_50/{joint}@16"),
            }
        )
    if rows:
        write_csv(TABLE_DIR / "conditioned_dual_50k_checkpoint_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = ["ood_correct@16", "ood_joint@16", "depth50_correct@16", "depth50_joint@16"]
    summary = df.groupby(["eval_template", "checkpoint"], as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    ).sort_values(["eval_template", "checkpoint"])
    write_csv(TABLE_DIR / "conditioned_dual_50k_checkpoint_summary.csv", summary.to_dict("records"))
    return summary


def plot_conditioned_50k_convergence(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    styles = {
        "conditioned_logic": ("#1f77b4", "conditioned logic"),
        "conditioned_nl": ("#d62728", "conditioned NL"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), sharex=True, sharey=True)
    for ax, (col, label) in zip(axes.ravel(), metrics, strict=True):
        for eval_template, (color, display) in styles.items():
            sub = summary[summary["eval_template"] == eval_template].dropna(subset=[col]).sort_values("checkpoint")
            if sub.empty:
                continue
            ax.plot(sub["checkpoint"], sub[col], marker="o", color=color, label=display)
        style_axes(ax, label)
        ax.set_xlabel("optimizer step")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "ablation_conditioned_dual_50k_convergence_train1to25")


def _summarize_dual_joint_passk(
    root: Path,
    regex: re.Pattern[str],
    field_names: list[str],
    summary_keys: list[str],
    table_prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*_passk.json")):
        if table_prefix == "trace_control_ablation" and _is_stale_rule_annotated_trace_repair(path, root):
            continue
        match = regex.match(path.name)
        if not match:
            continue
        group_map: dict[str, object] = dict(zip(field_names, match.groups(), strict=True))
        formal_joint_metric = "citation_free_joint_pass"
        if table_prefix == "trace_control_ablation":
            formal_joint_metric = "grounded_joint_pass"
        for numeric_field in ("train_max", "seed"):
            if numeric_field in group_map:
                group_map[numeric_field] = int(group_map[numeric_field])
        metrics = read_payload(path)["metrics"]
        rows.append(
            {
                **group_map,
                "ood_correct@16": metrics.get("synthetic_sampled/band_ood/correct_pass@16"),
                "ood_formal_joint@16": metrics.get(f"synthetic_sampled/band_ood/{formal_joint_metric}@16"),
                "ood_translated_joint@16": metrics.get("synthetic_sampled/band_ood/nl_logic_joint_pass@16"),
                "depth50_correct@16": metrics.get("synthetic_sampled/step_50/correct_pass@16"),
                "depth50_formal_joint@16": metrics.get(f"synthetic_sampled/step_50/{formal_joint_metric}@16"),
                "depth50_translated_joint@16": metrics.get("synthetic_sampled/step_50/nl_logic_joint_pass@16"),
            }
        )
    if rows:
        write_csv(TABLE_DIR / f"{table_prefix}_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "ood_correct@16",
        "ood_formal_joint@16",
        "ood_translated_joint@16",
        "depth50_correct@16",
        "depth50_formal_joint@16",
        "depth50_translated_joint@16",
    ]
    summary = df.groupby(summary_keys, as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    ).sort_values(summary_keys)
    write_csv(TABLE_DIR / f"{table_prefix}_summary.csv", summary.to_dict("records"))
    return summary


def _is_stale_rule_annotated_trace_repair(path: Path, root: Path) -> bool:
    stale_name = "sft_hfsa_ablate_rule_annotated_nl_train1to25_10k_seed3409_passk.json"
    if path.name != stale_name:
        return False
    repaired_seed = root / "sft_hfsa_ablate_rule_annotated_nl_train1to25_10k_seed3407_passk.json"
    return repaired_seed.exists() and path.stat().st_mtime < repaired_seed.stat().st_mtime


def plot_trace_control_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    df = summary.copy()
    template_order = {
        "main_logic": 0,
        "main_nl_exact": 1,
        "terse_nl": 2,
        "rule_annotated_nl": 3,
        "pseudocode": 4,
        "shuffled_logic": 5,
        "invalid_logic": 6,
        "shuffled_nl": 7,
    }
    df["order"] = df["template"].map(template_order).fillna(99)
    df = df.sort_values("order")
    labels = [str(value).replace("_", "\n") for value in df["template"]]
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_formal_joint@16", "OOD formal joint@16"),
        ("ood_translated_joint@16", "OOD translated joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharey=True)
    colors = [
        COLORS["logic"]
        if str(template) == "main_logic"
        else COLORS["nl_exact"]
        if str(template) == "main_nl_exact"
        else "#4c78a8"
        for template in df["template"]
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        values = [float(value) if isinstance(value, Number) and not pd.isna(value) else 0.0 for value in df[col]]
        ax.bar(range(len(df)), values, color=colors)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        style_axes(ax, title)
    save(fig, "ablation_trace_controls_summary")


def plot_hybrid_order_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_formal_joint@16", "OOD formal joint@16"),
        ("ood_translated_joint@16", "OOD translated joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
    ]
    styles = {
        "main_logic": ("#1f77b4", "-", "main logic"),
        "formal_think": ("#d62728", "-", "formal then NL"),
        "think_formal": ("#9467bd", "-", "NL then formal"),
        "main_nl_exact": ("#ff7f0e", "-", "main NL exact"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        for mode, (color, linestyle, label) in styles.items():
            sub = summary[summary["mode"] == mode].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=[col]).sort_values("train_max")
            if sub.empty:
                continue
            ax.plot(sub["train_max"], sub[col], marker="o", color=color, linestyle=linestyle, label=label)
        style_axes(ax, title)
        ax.set_xlabel("max train depth")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "ablation_hybrid_order_partial")


def summarize_paired_full_suite() -> pd.DataFrame:
    root = PASSK_ROOT / "paired_full_suite_sparse_20260528"
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*_passk.json")):
        match = PAIRED_FULL_RE.match(path.name)
        if not match:
            continue
        family, template, train_max, seed = match.groups()
        payload = read_payload(path)
        metrics = payload["metrics"]
        joint = "citation_free_joint_pass" if template == "logic" else "nl_logic_joint_pass"
        rows.append(
            {
                "family": family,
                "template": template,
                "train_max": int(train_max),
                "seed": int(seed),
                "train_correct@16": metrics.get("synthetic_sampled/band_train/correct_pass@16"),
                "train_joint@16": metrics.get(f"synthetic_sampled/band_train/{joint}@16"),
                "train_grounded_joint@16": metrics.get(
                    "synthetic_sampled/band_train/citation_free_grounded_joint_pass@16"
                ),
                "train_nl_parse@16": metrics.get("synthetic_sampled/band_train/nl_logic_parse_pass@16"),
                "ood_correct@16": metrics.get("synthetic_sampled/band_ood/correct_pass@16"),
                "ood_joint@16": metrics.get(f"synthetic_sampled/band_ood/{joint}@16"),
                "ood_grounded_joint@16": metrics.get(
                    "synthetic_sampled/band_ood/citation_free_grounded_joint_pass@16"
                ),
                "ood_nl_parse@16": metrics.get("synthetic_sampled/band_ood/nl_logic_parse_pass@16"),
                "depth50_correct@16": metrics.get("synthetic_sampled/step_50/correct_pass@16"),
                "depth50_joint@16": metrics.get(f"synthetic_sampled/step_50/{joint}@16"),
                "depth50_grounded_joint@16": metrics.get(
                    "synthetic_sampled/step_50/citation_free_grounded_joint_pass@16"
                ),
                "depth50_nl_parse@16": metrics.get("synthetic_sampled/step_50/nl_logic_parse_pass@16"),
            }
        )
    if rows:
        write_csv(TABLE_DIR / "paired_full_suite_partial_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "train_correct@16",
        "train_joint@16",
        "train_grounded_joint@16",
        "train_nl_parse@16",
        "ood_correct@16",
        "ood_joint@16",
        "ood_grounded_joint@16",
        "ood_nl_parse@16",
        "depth50_correct@16",
        "depth50_joint@16",
        "depth50_grounded_joint@16",
        "depth50_nl_parse@16",
    ]
    summary = df.groupby(["family", "template", "train_max"], as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    )
    summary = summary.sort_values(["family", "template", "train_max"])
    write_csv(TABLE_DIR / "paired_full_suite_partial_summary.csv", summary.to_dict("records"))
    return summary


def _metric_any(metrics: dict[str, float], names: list[str]) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _semantic_igsm_roots() -> list[Path]:
    candidates = [PASSK_ROOT / "paired_igsm_semantic_sparse_20260603"]
    hpcvault = os.environ.get("HPCVAULT")
    if hpcvault:
        candidates.append(
            Path(hpcvault) / "synthetic-RLVL" / "passk_eval" / "paired_igsm_semantic_sparse_20260603"
        )
    roots: list[Path] = []
    seen: set[Path] = set()
    for root in candidates:
        resolved = root.resolve() if root.exists() else root
        if resolved not in seen:
            seen.add(resolved)
            roots.append(root)
    return roots


def _passk_roots(subdir: str) -> list[Path]:
    candidates = [PASSK_ROOT / subdir]
    hpcvault = os.environ.get("HPCVAULT")
    if hpcvault:
        candidates.append(Path(hpcvault) / "synthetic-RLVL" / "passk_eval" / subdir)
    roots: list[Path] = []
    seen: set[Path] = set()
    for root in candidates:
        resolved = root.resolve() if root.exists() else root
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(root)
    return roots


def summarize_paired_igsm_semantic() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in _semantic_igsm_roots():
        if not root.exists():
            continue
        for path in sorted(root.glob("*_passk.json")):
            match = PAIRED_IGSM_SEMANTIC_RE.match(path.name)
            if not match:
                continue
            template, train_max, seed = match.groups()
            metrics = read_payload(path)["metrics"]
            valid_name = "citation_free_valid_pass" if template == "logic" else "nl_logic_parse_pass"
            joint_name = "citation_free_joint_pass" if template == "logic" else "nl_logic_joint_pass"
            rows.append(
                {
                    "template": template,
                    "train_max": int(train_max),
                    "seed": int(seed),
                    "ood_correct@16": _metric_any(
                        metrics,
                        [
                            "synthetic_sampled/band_hard_tail/correct_pass@16",
                            "synthetic_sampled/band_ood/correct_pass@16",
                        ],
                    ),
                    "ood_valid_or_parse@16": _metric_any(
                        metrics,
                        [
                            f"synthetic_sampled/band_hard_tail/{valid_name}@16",
                            f"synthetic_sampled/band_ood/{valid_name}@16",
                        ],
                    ),
                    "ood_joint@16": _metric_any(
                        metrics,
                        [
                            f"synthetic_sampled/band_hard_tail/{joint_name}@16",
                            f"synthetic_sampled/band_ood/{joint_name}@16",
                        ],
                    ),
                    "ood_grounded_joint@16": _metric_any(
                        metrics,
                        [
                            "synthetic_sampled/band_hard_tail/citation_free_grounded_joint_pass@16",
                            "synthetic_sampled/band_ood/citation_free_grounded_joint_pass@16",
                        ],
                    ),
                    "depth50_correct@16": metrics.get("synthetic_sampled/step_50/correct_pass@16"),
                    "depth50_valid_or_parse@16": metrics.get(f"synthetic_sampled/step_50/{valid_name}@16"),
                    "depth50_joint@16": metrics.get(f"synthetic_sampled/step_50/{joint_name}@16"),
                    "depth50_grounded_joint@16": metrics.get(
                        "synthetic_sampled/step_50/citation_free_grounded_joint_pass@16"
                    ),
                }
            )
    if rows:
        write_csv(TABLE_DIR / "paired_igsm_semantic_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "ood_correct@16",
        "ood_valid_or_parse@16",
        "ood_joint@16",
        "ood_grounded_joint@16",
        "depth50_correct@16",
        "depth50_valid_or_parse@16",
        "depth50_joint@16",
        "depth50_grounded_joint@16",
    ]
    summary = df.groupby(["template", "train_max"], as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    )
    summary = summary.sort_values(["template", "train_max"])
    write_csv(TABLE_DIR / "paired_igsm_semantic_summary.csv", summary.to_dict("records"))
    return summary


def plot_paired_igsm_semantic(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD hard-tail correct@16"),
        ("ood_valid_or_parse@16", "OOD valid/parse@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_valid_or_parse@16", "depth-50 valid/parse@16"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        for template, color in [("logic", COLORS["logic"]), ("nl_exact", COLORS["nl_exact"])]:
            sub = summary[summary["template"] == template].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=[col]).sort_values("train_max")
            if sub.empty:
                continue
            ax.plot(
                sub["train_max"],
                sub[col],
                marker="o",
                color=color,
                label=TEMPLATE_LABEL.get(template, template),
            )
        style_axes(ax, title)
        ax.set_xlabel("max train depth")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "paired_igsm_semantic_summary")


def _paired_joint_name(template: str) -> str:
    return "citation_free_joint_pass" if template == "logic" else "nl_logic_joint_pass"


def _paired_valid_or_parse_name(template: str) -> str:
    return "citation_free_valid_pass" if template == "logic" else "nl_logic_parse_pass"


def _metric_from_bands(metrics: dict[str, float], band: str, name: str, k: int = 16) -> float | None:
    return _metric_any(
        metrics,
        [
            f"synthetic_sampled/band_{band}/{name}@{k}",
            f"synthetic_sampled/{band}/{name}@{k}",
        ],
    )


def _metric_from_step(metrics: dict[str, float], depth: int, name: str, k: int = 16) -> float | None:
    return _metric_any(
        metrics,
        [
            f"synthetic_sampled/step_{depth}/{name}@{k}",
            f"synthetic_sampled/depth_{depth}/{name}@{k}",
        ],
    )


def summarize_active_paired_partials() -> pd.DataFrame:
    specs = [
        (
            "typed_maze",
            "typed maze",
            _passk_roots("paired_maze_typed_sparse_20260603"),
            PAIRED_MAZE_TYPED_RE,
            "formal cap 4096; NL cap 6144",
        ),
        (
            "hard_attribute_fresh",
            "hard attribute",
            _passk_roots("paired_attribute_constraints_hard_full_20260610"),
            PAIRED_HARD_ATTR_FRESH_RE,
            "formal cap 12288 except targeted row-1 recovery at 8192; NL cap 8192",
        ),
    ]
    rows: list[dict[str, object]] = []
    for family, display_family, roots, regex, cap_note in specs:
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*_passk.json")):
                match = regex.match(path.name)
                if not match:
                    continue
                template, train_max, seed = match.groups()
                metrics = read_payload(path)["metrics"]
                joint = _paired_joint_name(template)
                valid_or_parse = _paired_valid_or_parse_name(template)
                rows.append(
                    {
                        "family": family,
                        "display_family": display_family,
                        "template": template,
                        "train_max": int(train_max),
                        "seed": int(seed),
                        "train_correct@16": _metric_from_bands(metrics, "train", "correct_pass"),
                        "train_joint@16": _metric_from_bands(metrics, "train", joint),
                        "ood_correct@16": _metric_any(
                            metrics,
                            [
                                "synthetic_sampled/band_hard_tail/correct_pass@16",
                                "synthetic_sampled/band_ood/correct_pass@16",
                            ],
                        ),
                        "ood_valid_or_parse@16": _metric_any(
                            metrics,
                            [
                                f"synthetic_sampled/band_hard_tail/{valid_or_parse}@16",
                                f"synthetic_sampled/band_ood/{valid_or_parse}@16",
                            ],
                        ),
                        "ood_joint@16": _metric_any(
                            metrics,
                            [
                                f"synthetic_sampled/band_hard_tail/{joint}@16",
                                f"synthetic_sampled/band_ood/{joint}@16",
                            ],
                        ),
                        "ood_format@16": _metric_any(
                            metrics,
                            [
                                "synthetic_sampled/band_hard_tail/format_pass@16",
                                "synthetic_sampled/band_ood/format_pass@16",
                            ],
                        ),
                        "depth50_correct@16": _metric_from_step(metrics, 50, "correct_pass"),
                        "depth50_valid_or_parse@16": _metric_from_step(metrics, 50, valid_or_parse),
                        "depth50_joint@16": _metric_from_step(metrics, 50, joint),
                        "cap_note": cap_note,
                    }
                )
    if rows:
        write_csv(TABLE_DIR / "active_paired_partial_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "train_correct@16",
        "train_joint@16",
        "ood_correct@16",
        "ood_valid_or_parse@16",
        "ood_joint@16",
        "ood_format@16",
        "depth50_correct@16",
        "depth50_valid_or_parse@16",
        "depth50_joint@16",
    ]
    summary = df.groupby(["family", "display_family", "template", "train_max"], as_index=False).agg(
        n=("seed", "nunique"),
        cap_note=("cap_note", "first"),
        **{col: (col, "mean") for col in metric_cols},
    )
    summary = summary.sort_values(["family", "template", "train_max"])
    write_csv(TABLE_DIR / "active_paired_partial_summary.csv", summary.to_dict("records"))
    return summary


def plot_active_paired_partials(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    families = [("typed_maze", "typed maze"), ("hard_attribute_fresh", "hard attribute")]
    fig, axes = plt.subplots(len(families), len(metrics), figsize=(11.0, 6.0), sharex="row", sharey=True)
    for row_idx, (family, label) in enumerate(families):
        fam = summary[summary["family"] == family].copy()
        for col_idx, (col, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for template, color in [("logic", COLORS["logic"]), ("nl_exact", COLORS["nl_exact"])]:
                sub = fam[fam["template"] == template].dropna(subset=[col]).sort_values("train_max")
                if sub.empty:
                    continue
                ax.plot(sub["train_max"], sub[col], marker="o", color=color, label=TEMPLATE_LABEL.get(template, template))
                for _, point in sub.iterrows():
                    if int(point["n"]) < 3:
                        ax.annotate(
                            f"n={int(point['n'])}",
                            (point["train_max"], point[col]),
                            fontsize=7,
                            xytext=(3, 3),
                            textcoords="offset points",
                        )
            style_axes(ax, f"{label}: {title}")
            ax.set_xlabel("max train depth")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "active_paired_partial_summary")


def _batch_eval_joint_name(eval_condition: str) -> str:
    return "citation_free_joint_pass" if eval_condition in {"logic", "conditioned_logic"} else "nl_logic_joint_pass"


def _batch_eval_valid_or_parse_name(eval_condition: str) -> str:
    return "citation_free_valid_pass" if eval_condition in {"logic", "conditioned_logic"} else "nl_logic_parse_pass"


def summarize_hfsa_batch_size_partials() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in _passk_roots("hfsa_batch_size_ablation_20260603"):
        if not root.exists():
            continue
        for path in sorted(root.glob("*_passk.json")):
            match = HFSA_BATCH_SIZE_RE.match(path.name)
            if not match:
                continue
            batch_size, train_condition, train_max, seed, eval_condition = match.groups()
            metrics = read_payload(path)["metrics"]
            joint = _batch_eval_joint_name(eval_condition)
            valid_or_parse = _batch_eval_valid_or_parse_name(eval_condition)
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "train_condition": train_condition,
                    "eval_condition": eval_condition,
                    "train_max": int(train_max),
                    "seed": int(seed),
                    "ood_correct@16": _metric_any(
                        metrics,
                        [
                            "synthetic_sampled/band_hard_tail/correct_pass@16",
                            "synthetic_sampled/band_ood/correct_pass@16",
                        ],
                    ),
                    "ood_valid_or_parse@16": _metric_any(
                        metrics,
                        [
                            f"synthetic_sampled/band_hard_tail/{valid_or_parse}@16",
                            f"synthetic_sampled/band_ood/{valid_or_parse}@16",
                        ],
                    ),
                    "ood_joint@16": _metric_any(
                        metrics,
                        [
                            f"synthetic_sampled/band_hard_tail/{joint}@16",
                            f"synthetic_sampled/band_ood/{joint}@16",
                        ],
                    ),
                    "ood_format@16": _metric_any(
                        metrics,
                        [
                            "synthetic_sampled/band_hard_tail/format_pass@16",
                            "synthetic_sampled/band_ood/format_pass@16",
                        ],
                    ),
                    "depth50_correct@16": _metric_from_step(metrics, 50, "correct_pass"),
                    "depth50_valid_or_parse@16": _metric_from_step(metrics, 50, valid_or_parse),
                    "depth50_joint@16": _metric_from_step(metrics, 50, joint),
                }
            )
    if rows:
        write_csv(TABLE_DIR / "hfsa_batch_size_ablation_partial_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "ood_correct@16",
        "ood_valid_or_parse@16",
        "ood_joint@16",
        "ood_format@16",
        "depth50_correct@16",
        "depth50_valid_or_parse@16",
        "depth50_joint@16",
    ]
    summary = df.groupby(["train_condition", "eval_condition", "train_max", "batch_size"], as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    )
    summary = summary.sort_values(["eval_condition", "batch_size"])
    write_csv(TABLE_DIR / "hfsa_batch_size_ablation_partial_summary.csv", summary.to_dict("records"))
    return summary


def build_hfsa_batch_size_diagnostic_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    singletons = {
        ("conditioned_dual", "conditioned_logic"): ("logic", "logic"),
        ("conditioned_dual", "conditioned_nl"): ("nl_exact", "nl_exact"),
    }
    keyed = {
        (str(r["train_condition"]), str(r["eval_condition"]), int(r["batch_size"])): r
        for _, r in summary.iterrows()
    }
    for (train_condition, eval_condition), group in summary.groupby(["train_condition", "eval_condition"], sort=True):
        group = group.sort_values("batch_size")
        ood_best = group.dropna(subset=["ood_joint@16"]).sort_values(
            ["ood_joint@16", "batch_size"], ascending=[False, True]
        )
        d50_best = group.dropna(subset=["depth50_joint@16"]).sort_values(
            ["depth50_joint@16", "batch_size"], ascending=[False, True]
        )
        row: dict[str, object] = {
            "train_condition": train_condition,
            "eval_condition": eval_condition,
            "best_ood_joint_bsz": int(ood_best.iloc[0]["batch_size"]) if not ood_best.empty else "",
            "best_ood_joint@16": float(ood_best.iloc[0]["ood_joint@16"]) if not ood_best.empty else float("nan"),
            "best_depth50_joint_bsz": int(d50_best.iloc[0]["batch_size"]) if not d50_best.empty else "",
            "best_depth50_joint@16": float(d50_best.iloc[0]["depth50_joint@16"]) if not d50_best.empty else float("nan"),
            "ood_joint_range": float(group["ood_joint@16"].max() - group["ood_joint@16"].min()),
            "depth50_joint_range": float(group["depth50_joint@16"].max() - group["depth50_joint@16"].min()),
        }
        baseline_key = singletons.get((train_condition, eval_condition))
        if baseline_key:
            deltas = []
            for batch_size in sorted(group["batch_size"].astype(int).unique()):
                current = keyed.get((train_condition, eval_condition, int(batch_size)))
                baseline = keyed.get((baseline_key[0], baseline_key[1], int(batch_size)))
                if current is None or baseline is None:
                    continue
                deltas.append(float(current["ood_joint@16"]) - float(baseline["ood_joint@16"]))
            row["mean_ood_joint_delta_vs_single"] = float(mean(deltas)) if deltas else float("nan")
            row["best_ood_joint_delta_vs_single"] = (
                float(ood_best.iloc[0]["ood_joint@16"])
                - float(
                    keyed[(baseline_key[0], baseline_key[1], int(ood_best.iloc[0]["batch_size"]))]["ood_joint@16"]
                )
                if not ood_best.empty
                and (baseline_key[0], baseline_key[1], int(ood_best.iloc[0]["batch_size"])) in keyed
                else float("nan")
            )
        else:
            row["mean_ood_joint_delta_vs_single"] = float("nan")
            row["best_ood_joint_delta_vs_single"] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["eval_condition", "train_condition"])
    write_csv(TABLE_DIR / "hfsa_batch_size_ablation_diagnostics.csv", out.to_dict("records"))
    return out


def plot_hfsa_batch_size_partials(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
        ("depth50_joint@16", "depth-50 joint@16"),
    ]
    styles = {
        "logic": ("#1f77b4", "logic"),
        "nl_exact": ("#d62728", "NL exact"),
        "conditioned_logic": ("#1f77b4", "conditioned logic"),
        "conditioned_nl": ("#d62728", "conditioned NL"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.2), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes.ravel(), metrics, strict=True):
        for condition, (color, label) in styles.items():
            sub = summary[summary["eval_condition"] == condition].dropna(subset=[col]).sort_values("batch_size")
            if sub.empty:
                continue
            linestyle = "--" if condition.startswith("conditioned") else "-"
            ax.plot(sub["batch_size"], sub[col], marker="o", color=color, linestyle=linestyle, label=label)
        style_axes(ax, title)
        ax.set_xlabel("effective batch size")
        ax.set_xticks([2, 4, 8, 16])
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "hfsa_batch_size_ablation_partial")


def plot_hfsa_batch_size_conditioned_deltas(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    keyed = {
        (str(r["train_condition"]), str(r["eval_condition"]), int(r["batch_size"])): r
        for _, r in summary.iterrows()
    }
    pairs = [
        ("conditioned_dual", "conditioned_logic", "logic", "logic", "conditioned logic - logic", "#1f77b4"),
        ("conditioned_dual", "conditioned_nl", "nl_exact", "nl_exact", "conditioned NL - NL", "#d62728"),
    ]
    metrics = [("ood_joint@16", "OOD joint@16 delta"), ("depth50_joint@16", "depth-50 joint@16 delta")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharex=True, sharey=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for train, eval_condition, base_train, base_eval, label, color in pairs:
            xs: list[int] = []
            ys: list[float] = []
            for batch_size in [2, 4, 8, 16]:
                current = keyed.get((train, eval_condition, batch_size))
                baseline = keyed.get((base_train, base_eval, batch_size))
                if current is None or baseline is None:
                    continue
                xs.append(batch_size)
                ys.append(float(current[metric]) - float(baseline[metric]))
            if xs:
                ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        style_axes(ax, title)
        ax.set_xlabel("effective batch size")
        ax.set_xticks([2, 4, 8, 16])
    axes[0].set_ylabel("conditioned minus single-modality")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "hfsa_batch_size_conditioned_delta")


def plot_paired_full_suite_partial(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    igsm = summary[summary["family"] == "official_igsm"].copy()
    if igsm.empty:
        return
    metrics = [
        ("ood_correct@16", "OOD correct@16"),
        ("ood_joint@16", "OOD template-valid joint@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
    for ax, (col, title) in zip(axes, metrics, strict=True):
        for template, color in [("logic", COLORS["logic"]), ("nl_exact", COLORS["nl_exact"])]:
            sub = igsm[igsm["template"] == template].sort_values("train_max")
            if sub.empty:
                continue
            ax.plot(sub["train_max"], sub[col], marker="o", color=color, label=TEMPLATE_LABEL.get(template, template))
            for _, row in sub.iterrows():
                if int(row["n"]) < 3:
                    ax.annotate(f"n={int(row['n'])}", (row["train_max"], row[col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        style_axes(ax, title)
        ax.set_xlabel("max train depth")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "paired_full_suite_official_igsm_partial")


def plot_paired_full_suite_family_partial(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metrics = [
        ("train_correct@16", "train correct@16"),
        ("ood_correct@16", "OOD correct@16"),
        ("depth50_correct@16", "depth-50 correct@16"),
    ]
    families = [
        ("official_igsm", "iGSM"),
        ("maze_navigation", "maze"),
        ("attribute_constraints_hard", "attribute constraints"),
    ]
    fig, axes = plt.subplots(len(families), len(metrics), figsize=(10.8, 7.4), sharex=True, sharey="col")
    for row_idx, (family, family_label) in enumerate(families):
        fam = summary[summary["family"] == family].copy()
        for col_idx, (metric_col, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for template, color in [("logic", COLORS["logic"]), ("nl_exact", COLORS["nl_exact"])]:
                sub = fam[fam["template"] == template].dropna(subset=[metric_col]).sort_values("train_max")
                if sub.empty:
                    continue
                ax.plot(
                    sub["train_max"],
                    sub[metric_col],
                    marker="o",
                    color=color,
                    label=TEMPLATE_LABEL.get(template, template),
                )
                for _, point in sub.iterrows():
                    if int(point["n"]) < 3:
                        ax.annotate(
                            f"n={int(point['n'])}",
                            (point["train_max"], point[metric_col]),
                            fontsize=7,
                            xytext=(3, 3),
                            textcoords="offset points",
                        )
            style_axes(ax, f"{family_label}: {title}")
            if row_idx == len(families) - 1:
                ax.set_xlabel("max train depth")
            if col_idx == 0:
                ax.set_ylabel("pass@16")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "paired_full_suite_family_partial")


def build_experiment_artifact_status() -> pd.DataFrame:
    runs_root = WORK_ROOT / "synthetic-RLVL" / "runs"
    hpcvault = os.environ.get("HPCVAULT")
    extra_runs_root = Path(hpcvault) / "synthetic-RLVL" / "runs" if hpcvault else None
    extra_passk_root = Path(hpcvault) / "synthetic-RLVL" / "passk_eval" if hpcvault else None

    def has_final(run_name: str) -> bool:
        candidates = [runs_root / run_name / "final" / "adapter_config.json"]
        if extra_runs_root is not None:
            candidates.append(extra_runs_root / run_name / "final" / "adapter_config.json")
        return any(path.exists() for path in candidates)

    def count_jsons(root: Path | None, pattern: str = "*_passk.json") -> int:
        return len(list(root.glob(pattern))) if root is not None and root.exists() else 0

    rows: list[dict[str, object]] = []
    paired_eval_root = PASSK_ROOT / "paired_full_suite_sparse_20260528"
    for family, label in [
        ("official_igsm", "iGSM"),
        ("maze_navigation", "maze"),
        ("attribute_constraints_hard", "attribute constraints"),
    ]:
        expected = 30
        sft_done = 0
        for train_max in [5, 10, 15, 20, 25]:
            for template in ["logic", "nl_exact"]:
                for seed in [3407, 3408, 3409]:
                    run_name = f"sft_paired_full_{family}_{template}_train1to{train_max}_10k_seed{seed}"
                    sft_done += int(has_final(run_name))
        eval_done = count_jsons(paired_eval_root, f"sft_paired_full_{family}_*_passk.json")
        rows.append(
            {
                "experiment": label,
                "scope": "paired-family full suite",
                "sft_done": sft_done,
                "sft_expected": expected,
                "eval_done": eval_done,
                "eval_expected": expected,
                "status": "eval pending" if eval_done == 0 else ("complete" if eval_done == expected else "partial eval"),
            }
        )

    semantic_sft = 0
    for train_max in [5, 10, 15, 20, 25]:
        for template in ["logic", "nl_exact"]:
            for seed in [3407, 3408, 3409]:
                semantic_sft += int(has_final(f"sft_paired_igsm_semantic_{template}_train1to{train_max}_10k_seed{seed}"))
    semantic_eval = sum(
        count_jsons(root, "sft_paired_igsm_semantic_*_passk.json") for root in _semantic_igsm_roots()
    )
    rows.append(
        {
            "experiment": "semantic iGSM",
            "scope": "semantic bare-symbol paired rerun",
            "sft_done": semantic_sft,
            "sft_expected": 30,
            "eval_done": semantic_eval,
            "eval_expected": 30,
            "status": "complete" if semantic_eval == 30 else ("eval pending" if semantic_eval == 0 else "partial eval"),
        }
    )

    typed_maze_root = (
        extra_passk_root / "paired_maze_typed_sparse_20260603" if extra_passk_root is not None else None
    )
    typed_maze_sft = 0
    for train_max in [5, 10, 15, 20, 25]:
        for template in ["logic", "nl_exact"]:
            for seed in [3407, 3408, 3409]:
                typed_maze_sft += int(has_final(f"sft_paired_maze_typed_{template}_train1to{train_max}_10k_seed{seed}"))
    typed_maze_eval = count_jsons(typed_maze_root, "sft_paired_maze_typed_*_passk.json")
    rows.append(
        {
            "experiment": "typed maze",
            "scope": "typed-symbol paired rerun",
            "sft_done": typed_maze_sft,
            "sft_expected": 30,
            "eval_done": typed_maze_eval,
            "eval_expected": 30,
            "status": "complete" if typed_maze_eval == 30 else ("eval running" if typed_maze_eval == 0 and typed_maze_sft == 30 else "partial eval"),
        }
    )

    hard_attr_root = (
        extra_passk_root / "paired_attribute_constraints_hard_full_20260610"
        if extra_passk_root is not None
        else None
    )
    hard_attr_eval = count_jsons(hard_attr_root, "sft_paired_full_attribute_constraints_hard_*_passk.json")
    rows.append(
        {
            "experiment": "hard attribute fresh",
            "scope": "fresh hard-attribute-only eval",
            "sft_done": 30,
            "sft_expected": 30,
            "eval_done": hard_attr_eval,
            "eval_expected": 30,
            "status": "complete" if hard_attr_eval == 30 else ("eval running" if hard_attr_eval == 0 else "partial eval"),
        }
    )

    for name, scope, expected, root in [
        ("trace controls", "six train-1-to-25 perturbations", 18, PASSK_ROOT / "hfsa_ablation_trace_controls_20260525"),
        ("shortcut-rate", "rates 0.3/0.5/0.8, logic and NL", 18, PASSK_ROOT / "hfsa_shortcut_rate_ablation_20260525"),
        ("hybrid order", "formal-then-NL and NL-then-formal full train-depth suite", 30, PASSK_ROOT / "hfsa_hybrid_order_full_20260525"),
        ("shortcut-kind", "position and initial-marker shortcuts", 24, PASSK_ROOT / "hfsa_shortcut_kind_ablation_20260529"),
        ("conditioned dual 10k", "conditioned formal/NL dual modality", 30, PASSK_ROOT / "hfsa_conditioned_dual_full_20260525"),
        ("conditioned dual 50k", "longer conditioned formal/NL dual modality", 30, PASSK_ROOT / "hfsa_conditioned_dual_50k_20260529"),
    ]:
        eval_done = len(list(root.glob("*_passk.json"))) if root.exists() else 0
        status = "complete" if eval_done == expected else ("not started" if eval_done == 0 else "partial eval")
        if name == "conditioned dual 50k" and eval_done == 0:
            ckpt_root = PASSK_ROOT / "hfsa_conditioned_dual_50k_intermediate_20260529"
            ckpt_done = len(list(ckpt_root.glob("*_passk.json"))) if ckpt_root.exists() else 0
            if ckpt_done:
                status = f"checkpoint partial ({ckpt_done} JSONs)"
        rows.append(
            {
                "experiment": name,
                "scope": scope,
                "sft_done": "",
                "sft_expected": "",
                "eval_done": eval_done,
                "eval_expected": expected,
                "status": status,
            }
        )

    batch_root = extra_passk_root / "hfsa_batch_size_ablation_20260603" if extra_passk_root is not None else None
    batch_sft = 0
    for bsz in [2, 4, 8, 16]:
        for template in ["logic", "nl_exact", "conditioned_dual"]:
            batch_sft += int(has_final(f"sft_hfsa_batch_bsz{bsz}_{template}_train1to20_10k_seed3407"))
    batch_eval = count_jsons(batch_root, "*_passk.json")
    rows.append(
        {
            "experiment": "batch-size",
            "scope": "logic/NL/conditioned dual bsz 2/4/8/16",
            "sft_done": batch_sft,
            "sft_expected": 12,
            "eval_done": batch_eval,
            "eval_expected": 16,
            "status": "complete" if batch_eval == 16 else ("SFT recovery running" if batch_sft < 12 else ("eval pending" if batch_eval == 0 else "partial eval")),
        }
    )

    df = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "active_experiment_artifact_status.csv", df.to_dict("records"))
    return df


def build_token_budget_comparison_table(main_summary: pd.DataFrame, token_budget_summary: pd.DataFrame) -> pd.DataFrame:
    if main_summary.empty and token_budget_summary.empty:
        return pd.DataFrame()
    metrics = ["ood_correct@16", "ood_joint@16", "depth50_correct@16", "depth50_joint@16"]
    baseline = main_summary[
        (main_summary.get("size", "") == "")
        & (main_summary.get("template") == "logic")
        & (main_summary.get("train_max") == 25)
    ]
    baseline_values = baseline.iloc[0].to_dict() if not baseline.empty else {}
    rows: list[dict[str, object]] = []

    main_pair = main_summary[(main_summary.get("size", "") == "") & (main_summary.get("train_max") == 25)]
    for _, row in main_pair.sort_values("template").iterrows():
        item: dict[str, object] = {
            "condition": f"main {row['template']}",
            "steps": 10000,
            "n": row.get("n"),
        }
        for col in metrics:
            value = row.get(col)
            item[col] = value
            base = baseline_values.get(col)
            item[f"delta_{col}"] = value - base if isinstance(value, Number) and isinstance(base, Number) else None
        rows.append(item)

    for _, row in token_budget_summary.sort_values(["template", "steps"]).iterrows():
        item = {
            "condition": f"same-token {row['template']}",
            "steps": row.get("steps"),
            "n": row.get("n"),
        }
        for col in metrics:
            value = row.get(col)
            item[col] = value
            base = baseline_values.get(col)
            item[f"delta_{col}"] = value - base if isinstance(value, Number) and isinstance(base, Number) else None
        rows.append(item)

    df = pd.DataFrame(rows)
    if not df.empty:
        write_csv(TABLE_DIR / "same_target_token_budget_vs_main_logic.csv", df.to_dict("records"))
    return df


def build_token_budget_exposure_table() -> pd.DataFrame:
    logic_target_ratio_audit = 1038
    nl_target_ratio_audit = 1454
    logic_total_report_audit = 2587
    nl_total_report_audit = 3008
    nl_target_matched_steps = 7140
    nl_total_matched_steps = round(10000 * logic_total_report_audit / nl_total_report_audit)
    rows = [
        {
            "condition": "main logic baseline",
            "status": "result exists",
            "steps": 10000,
            "target_tok_per_ex": logic_target_ratio_audit,
            "total_tok_per_ex": logic_total_report_audit,
        },
        {
            "condition": "main NL baseline",
            "status": "result exists",
            "steps": 10000,
            "target_tok_per_ex": nl_target_ratio_audit,
            "total_tok_per_ex": nl_total_report_audit,
        },
        {
            "condition": "same-target logic control",
            "status": "result exists",
            "steps": 10000,
            "target_tok_per_ex": logic_target_ratio_audit,
            "total_tok_per_ex": logic_total_report_audit,
        },
        {
            "condition": "same-target NL",
            "status": "result exists",
            "steps": nl_target_matched_steps,
            "target_tok_per_ex": nl_target_ratio_audit,
            "total_tok_per_ex": nl_total_report_audit,
        },
        {
            "condition": "total-token NL match",
            "status": "not run",
            "steps": nl_total_matched_steps,
            "target_tok_per_ex": nl_target_ratio_audit,
            "total_tok_per_ex": nl_total_report_audit,
        },
    ]
    for row in rows:
        row["target_exposure_vs_logic"] = row["steps"] * row["target_tok_per_ex"] / (10000 * logic_target_ratio_audit)
        row["total_exposure_vs_logic"] = row["steps"] * row["total_tok_per_ex"] / (10000 * logic_total_report_audit)
    df = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "same_token_budget_exposure_accounting.csv", df.to_dict("records"))
    return df


def build_symbol_padded_token_match_table() -> pd.DataFrame:
    raw_parts = []
    for path in [
        TABLE_DIR / "logic_symbol_padded_length_audit_512.csv",
        TABLE_DIR / "logic_wordified_length_audit_512.csv",
    ]:
        if path.exists():
            raw_parts.append(pd.read_csv(path))
    if not raw_parts:
        return pd.DataFrame()
    raw = pd.concat(raw_parts, ignore_index=True)
    raw = raw.groupby("template", as_index=False).agg(
        n=("n", "max"),
        target_mean=("target_mean", "mean"),
        target_p95=("target_p95", "mean"),
        total_mean=("total_mean", "mean"),
        total_p95=("total_p95", "mean"),
        truncation_rate_at_max_length=("truncation_rate_at_max_length", "mean"),
    )
    labels = {
        "logic": "main logic",
        "logic_symbol_padded": "symbol-padded logic",
        "logic_wordified": "wordified logic",
        "nl_exact": "main NL exact",
    }
    order = {"logic": 0, "logic_symbol_padded": 1, "logic_wordified": 2, "nl_exact": 3}
    rows: list[dict[str, object]] = []
    baseline = raw[raw["template"] == "nl_exact"]
    nl_target = float(baseline.iloc[0]["target_mean"]) if not baseline.empty else None
    nl_total = float(baseline.iloc[0]["total_mean"]) if not baseline.empty else None
    for _, row in raw.sort_values("template", key=lambda s: s.map(order)).iterrows():
        target = float(row["target_mean"])
        total = float(row["total_mean"])
        rows.append(
            {
                "condition": labels.get(str(row["template"]), str(row["template"])),
                "n": int(row["n"]),
                "target_mean": target,
                "target_p95": float(row["target_p95"]),
                "total_mean": total,
                "total_p95": float(row["total_p95"]),
                "target_vs_nl": (target / nl_target) if nl_target else None,
                "total_vs_nl": (total / nl_total) if nl_total else None,
                "truncation_rate": float(row["truncation_rate_at_max_length"]),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        write_csv(TABLE_DIR / "logic_symbol_padded_token_match.csv", df.to_dict("records"))
        write_csv(TABLE_DIR / "logic_length_control_token_match.csv", df.to_dict("records"))
    return df


def build_symbol_padded_eval_comparison_table(
    main_summary: pd.DataFrame,
    symbol_padded_summary: pd.DataFrame,
    wordified_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if main_summary.empty and symbol_padded_summary.empty and (wordified_summary is None or wordified_summary.empty):
        return pd.DataFrame()
    metrics = ["ood_correct@16", "ood_joint@16", "depth50_correct@16", "depth50_joint@16"]
    rows: list[dict[str, object]] = []
    main_pair = main_summary[(main_summary.get("size", "") == "") & (main_summary.get("train_max") == 25)]
    labels = {
        "logic": "main compact logic",
        "nl_exact": "main NL exact",
        "logic_symbol_padded": "symbol-padded logic",
        "logic_wordified": "wordified logic",
    }
    for _, row in main_pair.sort_values("template").iterrows():
        item: dict[str, object] = {"condition": labels.get(str(row["template"]), str(row["template"])), "n": row.get("n")}
        for col in metrics:
            item[col] = row.get(col)
        rows.append(item)
    for summary in [symbol_padded_summary, wordified_summary if wordified_summary is not None else pd.DataFrame()]:
        if summary.empty:
            continue
        for _, row in summary.sort_values("template").iterrows():
            item = {"condition": labels.get(str(row["template"]), str(row["template"])), "n": row.get("n")}
            for col in metrics:
                item[col] = row.get(col)
            rows.append(item)
    df = pd.DataFrame(rows)
    if not df.empty:
        write_csv(TABLE_DIR / "logic_symbol_padded_eval_vs_main.csv", df.to_dict("records"))
        write_csv(TABLE_DIR / "logic_length_control_eval_vs_main.csv", df.to_dict("records"))
    return df


ABLATION_EXAMPLE_SPECS = [
    ("main_logic", TemplateName.LOGIC, "Normal compact formal-logic baseline."),
    ("main_nl_exact", TemplateName.NL_EXACT, "Normal exact natural-language baseline."),
    ("terse_nl", TemplateName.TERSE_NL, "Natural-language trace with shorter proof wording."),
    ("rule_annotated_nl", TemplateName.RULE_ANNOTATED_NL, "Natural-language trace with explicit rule labels."),
    ("pseudocode", TemplateName.PSEUDOCODE, "Algorithm-like natural-language derivation trace."),
    ("shuffled_logic", TemplateName.SHUFFLED_LOGIC, "Formal proof lines shuffled as an order negative control."),
    ("invalid_logic", TemplateName.INVALID_LOGIC, "Formal-looking trace with invalid rule citations."),
    ("shuffled_nl", TemplateName.SHUFFLED_NL, "Natural-language proof lines shuffled as an order negative control."),
    ("symbol_padded_logic", TemplateName.LOGIC_SYMBOL_PADDED, "Formal atoms expanded into predicate-call syntax."),
    ("wordified_logic", TemplateName.LOGIC_WORDIFIED, "Formal atoms expanded with natural-word predicate names."),
]


def _hard_fsa_config(template: TemplateName, *, seed: int = 3407, min_step: int = 1, max_step: int = 25) -> TaskConfig:
    return TaskConfig(
        template=template,
        prefill=PrefillMode.NONE,
        distractor_ratio=0.0,
        train_steps=StepRange(min_step, max_step),
        val_steps=StepRange(min_step, max_step),
        seed=seed,
        difficulty="hard_fsa",
        branching_factor=3,
        shortcut_rate=0.0,
        require_unique_solution=True,
    )


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))])


def _extract_tag_block(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


SYNTAX_LEXEMES = [
    "<formal>",
    "</formal>",
    "<think>",
    "</think>",
    "<constants>",
    "</constants>",
    "<predicates>",
    "</predicates>",
    "<premises>",
    "</premises>",
    "<proof>",
    "</proof>",
    "<conclusion>",
    "</conclusion>",
    "<answer>",
    "</answer>",
    "->",
    "&",
    ";",
    "(",
    ")",
    "MP",
    "R",
    "step_",
    "derive",
    "using",
    "[rule:",
    "]",
    "Therefore",
    "Combining",
]


def _syntax_token_stats(tokenizer, target: str) -> tuple[int, int]:
    occurrences = 0
    tokens = 0
    for lexeme in SYNTAX_LEXEMES:
        count = target.count(lexeme)
        if not count:
            continue
        occurrences += count
        tokens += count * _token_count(tokenizer, lexeme)
    return occurrences, tokens


def build_ablation_training_examples() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, template, description in ABLATION_EXAMPLE_SPECS:
        cfg = _hard_fsa_config(template, min_step=2, max_step=2)
        sample = TaskBuilder(cfg).sample(0, train=True)
        rows.append(
            {
                "condition": label,
                "template": template.value,
                "depth": sample.depth,
                "description": description,
                "sequence": sample.prompt + sample.target,
            }
        )
    write_csv(TABLE_DIR / "ablation_training_sequence_examples.csv", rows)
    return rows


def build_ablation_token_audit(sample_count: int = 512) -> pd.DataFrame:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
    rows: list[dict[str, object]] = []
    for label, template, description in ABLATION_EXAMPLE_SPECS:
        cfg = _hard_fsa_config(template)
        builder = TaskBuilder(cfg)
        prompt_lengths: list[int] = []
        target_lengths: list[int] = []
        total_lengths: list[int] = []
        proof_lengths: list[int] = []
        syntax_occurrences: list[int] = []
        syntax_tokens: list[int] = []
        for index in range(sample_count):
            sample = builder.sample(index, train=True)
            prompt_len = _token_count(tokenizer, sample.prompt)
            target_len = _token_count(tokenizer, sample.target)
            proof_text = _extract_tag_block(sample.target, "proof")
            proof_len = _token_count(tokenizer, proof_text) if proof_text else 0
            syntax_occ, syntax_tok = _syntax_token_stats(tokenizer, sample.target)
            prompt_lengths.append(prompt_len)
            target_lengths.append(target_len)
            total_lengths.append(prompt_len + target_len + 1)
            proof_lengths.append(proof_len)
            syntax_occurrences.append(syntax_occ)
            syntax_tokens.append(syntax_tok)
        target_mean = mean(target_lengths)
        syntax_occ_mean = mean(syntax_occurrences)
        syntax_tok_mean = mean(syntax_tokens)
        rows.append(
            {
                "condition": label,
                "template": template.value,
                "n": sample_count,
                "prompt_mean": mean(prompt_lengths),
                "target_mean": target_mean,
                "target_p95": _p95(target_lengths),
                "total_mean": mean(total_lengths),
                "total_p95": _p95(total_lengths),
                "proof_mean": mean(proof_lengths),
                "syntax_occ_mean": syntax_occ_mean,
                "syntax_tok_mean": syntax_tok_mean,
                "tok_per_syntax_occ": syntax_tok_mean / syntax_occ_mean if syntax_occ_mean else 0.0,
                "syntax_tok_share": syntax_tok_mean / target_mean if target_mean else 0.0,
                "description": description,
            }
        )
    df = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "ablation_training_token_audit_512.csv", df.to_dict("records"))
    return df


def build_trace_control_with_baselines(main_summary: pd.DataFrame, trace_control_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not main_summary.empty:
        main_pair = main_summary[
            (main_summary.get("size", "") == "")
            & (main_summary.get("train_max") == 25)
            & (main_summary.get("template").isin(["logic", "nl_exact"]))
        ]
        labels = {"logic": "main_logic", "nl_exact": "main_nl_exact"}
        for _, row in main_pair.sort_values("template").iterrows():
            is_logic = row["template"] == "logic"
            rows.append(
                {
                    "template": labels.get(str(row["template"]), str(row["template"])),
                    "n": row.get("n"),
                    "ood_correct@16": row.get("ood_correct@16"),
                    "ood_formal_joint@16": row.get("ood_joint@16") if is_logic else "",
                    "ood_translated_joint@16": row.get("ood_joint@16") if not is_logic else "",
                    "depth50_correct@16": row.get("depth50_correct@16"),
                    "depth50_formal_joint@16": row.get("depth50_joint@16") if is_logic else "",
                    "depth50_translated_joint@16": row.get("depth50_joint@16") if not is_logic else "",
                }
            )
    if not trace_control_rows.empty:
        for _, row in trace_control_rows.iterrows():
            rows.append(row.to_dict())
    df = pd.DataFrame(rows)
    write_csv(TABLE_DIR / "trace_control_ablation_with_main_baselines.csv", df.to_dict("records"))
    return df


def build_hybrid_order_with_baselines(main_summary: pd.DataFrame, hybrid_order_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_max_values = {5, 10, 15, 20, 25}
    if not hybrid_order_rows.empty and "train_max" in hybrid_order_rows:
        train_max_values.update(int(value) for value in hybrid_order_rows["train_max"].dropna().tolist())
    if not main_summary.empty:
        main_rows = main_summary[
            (main_summary.get("size", "") == "")
            & (main_summary.get("template").isin(["logic", "nl_exact"]))
            & (main_summary.get("train_max").isin(sorted(train_max_values)))
        ]
        for _, row in main_rows.iterrows():
            is_logic = row["template"] == "logic"
            rows.append(
                {
                    "mode": "main_logic" if is_logic else "main_nl_exact",
                    "train_max": int(row["train_max"]),
                    "n": row.get("n"),
                    "ood_correct@16": row.get("ood_correct@16"),
                    "ood_formal_joint@16": row.get("ood_joint@16") if is_logic else "",
                    "ood_translated_joint@16": row.get("ood_joint@16") if not is_logic else "",
                    "depth50_correct@16": row.get("depth50_correct@16"),
                    "depth50_formal_joint@16": row.get("depth50_joint@16") if is_logic else "",
                    "depth50_translated_joint@16": row.get("depth50_joint@16") if not is_logic else "",
                }
            )
    if not hybrid_order_rows.empty:
        for _, row in hybrid_order_rows.iterrows():
            rows.append(row.to_dict())
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"main_logic": 0, "formal_think": 1, "think_formal": 2, "main_nl_exact": 3}
        df["mode_order"] = df["mode"].map(order).fillna(99)
        df = df.sort_values(["train_max", "mode_order"]).drop(columns=["mode_order"])
        write_csv(TABLE_DIR / "hybrid_order_with_main_baselines.csv", df.to_dict("records"))
    return df


def latex_ablation_examples(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No ablation examples generated."
    parts = [
        (
            "Each block below is a complete depth-2 SFT training sequence: prompt followed by target. "
            "The examples are generated deterministically from the same underlying HFSA item where possible, "
            "so differences are due to the trace template rather than a different task instance."
        )
    ]
    for row in rows:
        condition = str(row["condition"]).replace("_", r"\_")
        description = str(row["description"]).replace("_", r"\_")
        sequence = str(row["sequence"]).rstrip()
        parts.append(
            "\n".join(
                [
                    rf"\paragraph{{{condition}.}} {description}",
                    r"\begin{verbatim}",
                    sequence,
                    r"\end{verbatim}",
                ]
            )
        )
    return "\n\n".join(parts)


def build_length_control_surface_examples() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, template, description in [
        ("compact_logic", TemplateName.LOGIC, "Compact baseline formal symbols."),
        ("symbol_padded_logic", TemplateName.LOGIC_SYMBOL_PADDED, "Predicate-call symbols with padded constants."),
        ("wordified_logic", TemplateName.LOGIC_WORDIFIED, "Predicate-call symbols using semantic attribute words."),
    ]:
        cfg = _hard_fsa_config(template, min_step=2, max_step=2)
        sample = TaskBuilder(cfg).sample(0, train=True)
        target = sample.target
        predicates = "\n".join(_extract_tag_block(target, "predicates").splitlines()[:5])
        premises = "\n".join(_extract_tag_block(target, "premises").splitlines()[:4])
        proof = "\n".join(_extract_tag_block(target, "proof").splitlines()[:5])
        rows.append(
            {
                "condition": label,
                "description": description,
                "predicates": predicates,
                "premises": premises,
                "proof": proof,
            }
        )
    write_csv(TABLE_DIR / "logic_length_control_surface_examples.csv", rows)
    return rows


def latex_length_control_surface_examples(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No length-control surface examples generated."
    parts = [
        (
            "The examples below show the same depth-2 training item under the three formal surfaces. "
            "They are all valid gold targets; the difference is surface representation and tokenizer burden, "
            "not a different underlying task."
        )
    ]
    for row in rows:
        condition = str(row["condition"]).replace("_", r"\_")
        description = str(row["description"]).replace("_", r"\_")
        snippet = "\n".join(
            [
                "predicates:",
                str(row["predicates"]),
                "",
                "premises:",
                str(row["premises"]),
                "",
                "proof:",
                str(row["proof"]),
            ]
        )
        parts.append(
            "\n".join(
                [
                    rf"\paragraph{{{condition}.}} {description}",
                    r"\begin{verbatim}",
                    snippet,
                    r"\end{verbatim}",
                ]
            )
        )
    return "\n\n".join(parts)


def clean_text(text: str, limit: int = 850) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return clean_text(match.group(1), 120)
    return clean_text(text, 120)


def sample_from(path: Path, predicate) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if predicate(item):
                return item
    return None


def build_sample_panels() -> list[dict[str, object]]:
    candidates: list[tuple[str, Path, object]] = [
        (
            "Tiny 200M logic OOD correct sample",
            PASSK_ROOT / "hfsa_tiny_llama_pretrain_sparse/pretrain_hfsa_llama3_200m_logic_train1to10_seed3407_samples.jsonl",
            lambda x: int(x.get("step", 0)) > 10 and float(x.get("correct", 0)) == 1.0,
        ),
        (
            "Tiny 200M NL OOD failure sample",
            PASSK_ROOT / "hfsa_tiny_llama_pretrain_sparse/pretrain_hfsa_llama3_200m_nl_exact_train1to10_seed3407_samples.jsonl",
            lambda x: int(x.get("step", 0)) > 10 and float(x.get("correct", 0)) == 0.0,
        ),
        (
            "OLMo-7B logic train1..25 depth-50 correct+valid",
            PASSK_ROOT / "hfsa_depth_scaling_sparse/sft_hfsa_depth_scaling_logic_train1to25_10k_seed3407_samples.jsonl",
            lambda x: int(x.get("step", 0)) == 50
            and float(x.get("correct", 0)) == 1.0
            and float(x.get("citation_free_valid", 0)) == 1.0,
        ),
        (
            "OLMo-7B NL train1..25 depth-50 sample",
            PASSK_ROOT / "hfsa_depth_scaling_sparse/sft_hfsa_depth_scaling_nl_exact_train1to25_10k_seed3407_samples.jsonl",
            lambda x: int(x.get("step", 0)) == 50,
        ),
    ]
    samples = []
    for label, path, predicate in candidates:
        item = sample_from(path, predicate)
        if item is None:
            continue
        samples.append(
            {
                "label": label,
                "step": item.get("step"),
                "gold": item.get("gold_answer"),
                "answer": extract_answer(str(item.get("generation", ""))),
                "correct": item.get("correct"),
                "valid": item.get("citation_free_valid", item.get("nl_logic_citation_free_valid")),
                "generation": clean_text(str(item.get("generation", ""))),
            }
        )
    write_csv(TABLE_DIR / "sample_generation_snippets.csv", samples)
    return samples


def plot_sample_panels(samples: list[dict[str, object]]) -> None:
    if not samples:
        return
    with PdfPages(FIG_DIR / "sample_generation_panels.pdf") as pdf:
        for idx, sample in enumerate(samples):
            fig = plt.figure(figsize=(10.5, 5.8))
            ax = fig.add_subplot(111)
            ax.axis("off")
            header = (
                f"{sample['label']} | depth={sample['step']} | gold={sample['gold']} | "
                f"extracted={sample['answer']} | correct={sample['correct']} | valid={sample['valid']}"
            )
            wrapped = "\n".join(textwrap.wrap(str(sample["generation"]), width=118))
            ax.text(0.01, 0.98, header, ha="left", va="top", fontsize=10, weight="bold")
            ax.text(0.01, 0.90, wrapped, ha="left", va="top", fontsize=7.5, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(FIG_DIR / f"sample_generation_{idx}.png", dpi=180)
            plt.close(fig)


def latex_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    max_rows: int | None = None,
    bold_columns: set[str] | None = None,
) -> str:
    if df.empty:
        return "No rows available."
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    bold_columns = bold_columns or set()
    best_values: dict[str, float] = {}
    for col in bold_columns:
        values = [
            float(value)
            for value in data[col].tolist()
            if isinstance(value, Number) and not isinstance(value, bool) and not pd.isna(value)
        ]
        if values and len(set(values)) > 1:
            best_values[col] = max(values)
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule"]
    lines.append(" & ".join(label for _, label in columns) + " \\\\")
    lines.append("\\midrule")
    integer_columns = {
        "n",
        "seed",
        "train_max",
        "checkpoint",
        "depth",
        "logic_target",
        "nl_target",
        "logic_total",
        "nl_total",
        "steps",
        "target_tok_per_ex",
        "total_tok_per_ex",
        "target_mean",
        "target_p95",
        "total_mean",
        "total_p95",
        "prompt_mean",
        "prompt_p95",
        "proof_mean",
        "syntax_occ_mean",
        "syntax_tok_mean",
        "sft_done",
        "sft_expected",
        "eval_done",
        "eval_expected",
    }
    for _, row in data.iterrows():
        vals = []
        for col, _ in columns:
            value = row.get(col, "")
            if isinstance(value, Number) and not isinstance(value, bool):
                if col in integer_columns and float(value).is_integer():
                    cell = str(int(value))
                else:
                    cell = f3(value)
                if col in best_values and abs(float(value) - best_values[col]) < 1e-12:
                    cell = f"\\textbf{{{cell}}}"
                vals.append(cell)
            else:
                vals.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def get_lm_eval_metric(results: dict, task: str, key: str) -> float | None:
    value = results.get(task, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def get_lm_eval_metric_any(results: dict, task: str, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = get_lm_eval_metric(results, task, key)
        if value is not None:
            return value
    return None


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)(?:\s*</answer>|$)", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")


def _clean_extracted_answer(text: str) -> str:
    text = re.sub(r"</?(?:think|formal|natural|proof|conclusion|answer)[^>]*>", " ", str(text), flags=re.IGNORECASE)
    return " ".join(text.strip().split())


def _extract_explicit_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(str(text))
    if matches:
        return _clean_extracted_answer(matches[-1])
    for marker in ("Final answer:", "final answer:", "Answer:", "answer:"):
        if marker in str(text):
            return _clean_extracted_answer(str(text).rsplit(marker, 1)[-1])
    return ""


def _canonical_number(text: str) -> str:
    value = str(text).strip().replace("$", "").replace(",", "")
    if re.fullmatch(r"-?[0-9]+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def _last_number(text: str) -> str:
    matches = _NUMBER_RE.findall(str(text))
    if not matches:
        return ""
    return _canonical_number(matches[-1])


def _gold_gsm8k_answer(doc: dict) -> str:
    answer = str(doc.get("answer", ""))
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    return answer.strip()


def _raw_sample_response(row: dict) -> str:
    resps = row.get("resps") or []
    if resps and isinstance(resps[0], list) and resps[0]:
        return str(resps[0][0])
    if resps:
        return str(resps[0])
    return ""


def recompute_gsm8k_from_samples(run_dir: Path) -> dict[str, float] | None:
    sample_files = sorted(run_dir.glob("**/samples_synthrlvl_gsm8k_tagged_*.jsonl"))
    if not sample_files:
        return None
    n = correct = tag_found = nonempty = 0
    with sample_files[-1].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = _raw_sample_response(row)
            extracted = _extract_explicit_answer(raw)
            pred = _last_number(extracted) or extracted.strip()
            gold = _canonical_number(_gold_gsm8k_answer(row.get("doc", {})))
            correct += int(pred == gold)
            tag_found += int(bool(_ANSWER_RE.search(raw)))
            nonempty += int(bool(extracted.strip()))
            n += 1
    if n == 0:
        return None
    return {
        "gsm8k_em": correct / n,
        "gsm8k_tag": tag_found / n,
        "gsm8k_explicit_nonempty": nonempty / n,
    }


def load_main_ood_summary() -> pd.DataFrame:
    root = LM_EVAL_ROOT / "ood_large_2026-05-25"
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = MAIN_OOD_RE.match(run_dir.name)
        if not match:
            continue
        result_files = sorted(run_dir.glob("**/results_*.json"))
        if not result_files:
            continue
        payload = json.loads(result_files[-1].read_text(encoding="utf-8"))
        results = payload.get("results", {})
        gsm8k = recompute_gsm8k_from_samples(run_dir) or {}
        rows.append(
            {
                "template": match.group(1),
                "train_max": int(match.group(2)),
                "seed": int(match.group(3)),
                "gsm8k_em": gsm8k.get("gsm8k_em")
                if gsm8k
                else get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "exact_match,none"),
                "gsm8k_tag": gsm8k.get("gsm8k_tag")
                if gsm8k
                else get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "tag_found,none"),
                "gsm8k_explicit_nonempty": gsm8k.get("gsm8k_explicit_nonempty"),
                "hotpot_f1": get_lm_eval_metric(
                    results, "synthrlvl_longbench_hotpotqa_tagged", "qa_f1_score,none"
                ),
                "hotpot_em": get_lm_eval_metric(
                    results, "synthrlvl_longbench_hotpotqa_tagged", "qa_exact_match,none"
                ),
                "twowiki_f1": get_lm_eval_metric(
                    results, "synthrlvl_longbench_2wikimqa_tagged", "qa_f1_score,none"
                ),
                "twowiki_em": get_lm_eval_metric(
                    results, "synthrlvl_longbench_2wikimqa_tagged", "qa_exact_match,none"
                ),
                "musique_f1": get_lm_eval_metric(
                    results, "synthrlvl_longbench_musique_tagged", "qa_f1_score,none"
                ),
                "musique_em": get_lm_eval_metric(
                    results, "synthrlvl_longbench_musique_tagged", "qa_exact_match,none"
                ),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "gsm8k_em",
        "gsm8k_tag",
        "gsm8k_explicit_nonempty",
        "hotpot_f1",
        "hotpot_em",
        "twowiki_f1",
        "twowiki_em",
        "musique_f1",
        "musique_em",
    ]
    df.to_csv(TABLE_DIR / "main_olmo7b_ood_lmeval_by_seed.csv", index=False, lineterminator="\n")
    return df.groupby(["template", "train_max"], as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    ).sort_values(["template", "train_max"])


def load_tiny_ood_summary() -> pd.DataFrame:
    root = LM_EVAL_ROOT / "ood_tiny_llama_2026-05-25"
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = TINY_RE.match(run_dir.name + "_passk.json")
        if not match:
            continue
        result_files = sorted(run_dir.glob("**/results_*.json"))
        if not result_files:
            continue
        payload = json.loads(result_files[-1].read_text(encoding="utf-8"))
        results = payload.get("results", {})
        gsm8k = recompute_gsm8k_from_samples(run_dir) or {}

        rows.append(
            {
                "size": match.group(1),
                "template": match.group(2),
                "seed": int(match.group(3)),
                "gsm8k_em": gsm8k.get("gsm8k_em")
                if gsm8k
                else get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "exact_match,none"),
                "gsm8k_tag": gsm8k.get("gsm8k_tag")
                if gsm8k
                else get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "tag_found,none"),
                "gsm8k_explicit_nonempty": gsm8k.get("gsm8k_explicit_nonempty"),
                "hotpot_f1": get_lm_eval_metric(
                    results, "synthrlvl_longbench_hotpotqa_tagged", "qa_f1_score,none"
                ),
                "hotpot_em": get_lm_eval_metric(
                    results, "synthrlvl_longbench_hotpotqa_tagged", "exact_match,none"
                ),
                "twowiki_f1": get_lm_eval_metric(
                    results, "synthrlvl_longbench_2wikimqa_tagged", "qa_f1_score,none"
                ),
                "twowiki_em": get_lm_eval_metric(
                    results, "synthrlvl_longbench_2wikimqa_tagged", "exact_match,none"
                ),
                "musique_f1": get_lm_eval_metric(
                    results, "synthrlvl_longbench_musique_tagged", "qa_f1_score,none"
                ),
                "musique_em": get_lm_eval_metric(
                    results, "synthrlvl_longbench_musique_tagged", "exact_match,none"
                ),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        metric_cols = [
            "gsm8k_em",
            "gsm8k_tag",
            "gsm8k_explicit_nonempty",
            "hotpot_f1",
            "hotpot_em",
            "twowiki_f1",
            "twowiki_em",
            "musique_f1",
            "musique_em",
        ]
        df.to_csv(TABLE_DIR / "tiny_llama_ood_lmeval_by_seed.csv", index=False, lineterminator="\n")
        df = df.groupby(["size", "template"], as_index=False).agg(
            n=("seed", "nunique"),
            **{col: (col, "mean") for col in metric_cols},
        )
        df["size_order"] = df["size"].map(SIZE_ORDER)
        df = df.sort_values(["size_order", "template"]).drop(columns=["size_order"])
    return df


def _latest_result_file(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("**/results_*.json"))
    return files[-1] if files else None


def _cot_bare_metrics(results: dict) -> dict[str, float | None]:
    return {
        "gsm8k_em": get_lm_eval_metric(results, "synthrlvl_gsm8k_cot_bare", "exact_match,none"),
        "gsm8k_tag": get_lm_eval_metric(results, "synthrlvl_gsm8k_cot_bare", "tag_found,none"),
        "hotpot_em": get_lm_eval_metric_any(
            results, "synthrlvl_longbench_hotpotqa_cot_bare", ("qa_exact_match,none", "exact_match,none")
        ),
        "hotpot_f1": get_lm_eval_metric(
            results, "synthrlvl_longbench_hotpotqa_cot_bare", "qa_f1_score,none"
        ),
        "twowiki_em": get_lm_eval_metric_any(
            results, "synthrlvl_longbench_2wikimqa_cot_bare", ("qa_exact_match,none", "exact_match,none")
        ),
        "twowiki_f1": get_lm_eval_metric(
            results, "synthrlvl_longbench_2wikimqa_cot_bare", "qa_f1_score,none"
        ),
        "musique_em": get_lm_eval_metric_any(
            results, "synthrlvl_longbench_musique_cot_bare", ("qa_exact_match,none", "exact_match,none")
        ),
        "musique_f1": get_lm_eval_metric(
            results, "synthrlvl_longbench_musique_cot_bare", "qa_f1_score,none"
        ),
    }


def _summarize_lm_eval_rows(rows: list[dict[str, object]], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "gsm8k_em",
        "gsm8k_tag",
        "hotpot_em",
        "hotpot_f1",
        "twowiki_em",
        "twowiki_f1",
        "musique_em",
        "musique_f1",
    ]
    return df.groupby(keys, as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    ).sort_values(keys)


def load_main_cot_bare_ood_summary() -> pd.DataFrame:
    root = LM_EVAL_ROOT / "ood_large_cot_bare_2026-05-27"
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = MAIN_OOD_RE.match(run_dir.name)
        if not match:
            continue
        result_file = _latest_result_file(run_dir)
        if result_file is None:
            continue
        results = json.loads(result_file.read_text(encoding="utf-8")).get("results", {})
        rows.append(
            {
                "template": match.group(1),
                "train_max": int(match.group(2)),
                "seed": int(match.group(3)),
                **_cot_bare_metrics(results),
            }
        )
    if rows:
        write_csv(TABLE_DIR / "main_olmo7b_cot_bare_ood_by_seed.csv", rows)
    summary = _summarize_lm_eval_rows(rows, ["template", "train_max"])
    if not summary.empty:
        write_csv(TABLE_DIR / "main_olmo7b_cot_bare_ood_summary.csv", summary.to_dict("records"))
    return summary


def load_tiny_cot_bare_ood_summary(root_name: str, regex: re.Pattern[str], table_prefix: str) -> pd.DataFrame:
    root = LM_EVAL_ROOT / root_name
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = regex.match(run_dir.name)
        if not match:
            continue
        result_file = _latest_result_file(run_dir)
        if result_file is None:
            continue
        results = json.loads(result_file.read_text(encoding="utf-8")).get("results", {})
        rows.append(
            {
                "size": match.group(1),
                "template": match.group(2),
                "seed": int(match.group(3)),
                **_cot_bare_metrics(results),
            }
        )
    if rows:
        write_csv(TABLE_DIR / f"{table_prefix}_by_seed.csv", rows)
        if table_prefix == "conditioned_dual":
            write_csv(TABLE_DIR / "conditioned_dual_partial_by_seed.csv", rows)
    summary = _summarize_lm_eval_rows(rows, ["size", "template"])
    if not summary.empty:
        summary["size_order"] = summary["size"].map(SIZE_ORDER)
        summary = summary.sort_values(["size_order", "template"]).drop(columns=["size_order"])
        write_csv(TABLE_DIR / f"{table_prefix}_summary.csv", summary.to_dict("records"))
    return summary


def load_olmo32_cot_bare_gsm8k() -> pd.DataFrame:
    root = LM_EVAL_ROOT / "ood_large_olmo32_gsm8k_cot_bare_2026-05-27"
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = OLMO32_BARE_RE.match(run_dir.name)
        if not match:
            continue
        result_file = _latest_result_file(run_dir)
        if result_file is None:
            continue
        results = json.loads(result_file.read_text(encoding="utf-8")).get("results", {})
        rows.append(
            {
                "template": match.group(1),
                "train_max": 20,
                "seed": int(match.group(2)),
                "gsm8k_em": get_lm_eval_metric(results, "synthrlvl_gsm8k_cot_bare", "exact_match,none"),
                "gsm8k_tag": get_lm_eval_metric(results, "synthrlvl_gsm8k_cot_bare", "tag_found,none"),
            }
        )
    df = pd.DataFrame(rows).sort_values(["template", "seed"]) if rows else pd.DataFrame()
    if not df.empty:
        write_csv(TABLE_DIR / "olmo32_cot_bare_gsm8k.csv", df.to_dict("records"))
    return df


def _joint_key(label: str) -> str:
    return (
        "citation_free_joint_pass"
        if label in {"logic", "conditioned_logic", "logic_symbol_padded", "logic_wordified"}
        else "nl_logic_joint_pass"
    )


def _summarize_passk_ablation(
    root: Path,
    regex: re.Pattern[str],
    field_names: list[str],
    summary_keys: list[str],
    table_prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("*_passk.json")):
        match = regex.match(path.name)
        if not match:
            continue
        groups = list(match.groups())
        payload = read_payload(path)
        group_map = dict(zip(field_names, groups, strict=True))
        for numeric_field in ("train_max", "seed", "steps"):
            if numeric_field in group_map:
                group_map[numeric_field] = int(group_map[numeric_field])
        template_label = group_map.get("template") or group_map.get("eval_template") or "logic"
        joint = _joint_key(str(template_label))
        rows.append(
            {
                **group_map,
                "ood_correct@16": payload["metrics"].get("synthetic_sampled/band_ood/correct_pass@16"),
                "ood_joint@16": payload["metrics"].get(f"synthetic_sampled/band_ood/{joint}@16"),
                "depth50_correct@16": payload["metrics"].get("synthetic_sampled/step_50/correct_pass@16"),
                "depth50_joint@16": payload["metrics"].get(f"synthetic_sampled/step_50/{joint}@16"),
            }
        )
    if rows:
        write_csv(TABLE_DIR / f"{table_prefix}_by_seed.csv", rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = ["ood_correct@16", "ood_joint@16", "depth50_correct@16", "depth50_joint@16"]
    summary = df.groupby(summary_keys, as_index=False).agg(
        n=("seed", "nunique"),
        **{col: (col, "mean") for col in metric_cols},
    ).sort_values(summary_keys)
    write_csv(TABLE_DIR / f"{table_prefix}_summary.csv", summary.to_dict("records"))
    if table_prefix == "conditioned_dual":
        write_csv(TABLE_DIR / "conditioned_dual_partial_summary.csv", summary.to_dict("records"))
    return summary


def load_ablation_summaries() -> dict[str, pd.DataFrame]:
    return {
        "token_budget": _summarize_passk_ablation(
            PASSK_ROOT / "hfsa_same_target_token_budget_20260525",
            TOKBUDGET_RE,
            ["template", "steps", "seed"],
            ["template", "steps"],
            "same_target_token_budget",
        ),
        "shortcut": _summarize_passk_ablation(
            PASSK_ROOT / "hfsa_shortcut_rate_ablation_20260525",
            SHORTCUT_RE,
            ["template", "shortcut_rate", "seed"],
            ["template", "shortcut_rate"],
            "shortcut_rate_ablation",
        ),
        "shortcut_kind": _summarize_passk_ablation(
            PASSK_ROOT / "hfsa_shortcut_kind_ablation_20260529",
            SHORTCUT_KIND_RE,
            ["shortcut_kind", "template", "shortcut_rate", "seed"],
            ["shortcut_kind", "template", "shortcut_rate"],
            "shortcut_kind_ablation",
        ),
        "trace_control": _summarize_dual_joint_passk(
            PASSK_ROOT / "hfsa_ablation_trace_controls_20260525",
            TRACE_CONTROL_RE,
            ["template", "seed"],
            ["template"],
            "trace_control_ablation",
        ),
        "hybrid_order": _summarize_dual_joint_passk(
            PASSK_ROOT / "hfsa_hybrid_order_full_20260525",
            HYBRID_ORDER_RE,
            ["mode", "train_max", "seed"],
            ["mode", "train_max"],
            "hybrid_order_ablation",
        ),
        "conditioned": _summarize_passk_ablation(
            PASSK_ROOT / "hfsa_conditioned_dual_full_20260525",
            CONDITIONED_RE,
            ["train_max", "seed", "eval_template"],
            ["train_max", "eval_template"],
            "conditioned_dual",
        ),
        "conditioned_50k": _summarize_passk_ablation(
            PASSK_ROOT / "hfsa_conditioned_dual_50k_20260529",
            CONDITIONED_50K_RE,
            ["train_max", "seed", "eval_template"],
            ["train_max", "eval_template"],
            "conditioned_dual_50k",
        ),
        "symbol_padded": _summarize_passk_ablation(
            PASSK_ROOT / "hfsa_logic_symbol_padded_20260528",
            SYMBOL_PADDED_RE,
            ["template", "seed"],
            ["template"],
            "logic_symbol_padded_eval",
        ),
        "wordified": _summarize_passk_ablation(
            PASSK_ROOT / "hfsa_logic_wordified_20260529",
            WORDIFIED_RE,
            ["template", "seed"],
            ["template"],
            "logic_wordified_eval",
        ),
    }


COT_BARE_SAMPLE_PATTERNS = {
    "gsm8k": "samples_synthrlvl_gsm8k_cot_bare_*.jsonl",
    "hotpotqa": "samples_synthrlvl_longbench_hotpotqa_cot_bare_*.jsonl",
    "2wikimqa": "samples_synthrlvl_longbench_2wikimqa_cot_bare_*.jsonl",
    "musique": "samples_synthrlvl_longbench_musique_cot_bare_*.jsonl",
}


def _load_sample_file(run_dir: Path, pattern: str) -> dict[int, dict[str, object]]:
    files = sorted(run_dir.glob(f"**/{pattern}"))
    if not files:
        return {}
    rows: dict[int, dict[str, object]] = {}
    with files[-1].open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["doc_id"])] = row
    return rows


def _sample_question(row: dict[str, object]) -> str:
    doc = row.get("doc") or {}
    if isinstance(doc, dict):
        return " ".join(str(doc.get("question", "")).split())
    return ""


def _sample_gold(row: dict[str, object]) -> str:
    doc = row.get("doc") or {}
    if not isinstance(doc, dict):
        return ""
    if "answers" in doc:
        return ", ".join(str(item) for item in doc.get("answers", []))
    answer = str(doc.get("answer", row.get("target", "")))
    if "####" in answer:
        answer = answer.rsplit("####", 1)[-1]
    return " ".join(answer.split())


def _sample_metric(row: dict[str, object], task: str) -> float:
    if task == "gsm8k":
        return float(row.get("exact_match", 0.0) or 0.0)
    return float(row.get("qa_f1_score", row.get("score", 0.0)) or 0.0)


def _clip_for_report(text: str, limit: int = 900) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2].rstrip()
    tail = text[-limit // 2 :].lstrip()
    return f"{head}\n...[truncated]...\n{tail}"


def build_cot_bare_generation_examples() -> list[dict[str, object]]:
    logic_root = (
        LM_EVAL_ROOT
        / "ood_large_cot_bare_2026-05-27"
        / "sft_hfsa_depth_scaling_logic_train1to25_10k_seed3407"
    )
    nl_root = (
        LM_EVAL_ROOT
        / "ood_large_cot_bare_2026-05-27"
        / "sft_hfsa_depth_scaling_nl_exact_train1to25_10k_seed3407"
    )
    rows: list[dict[str, object]] = []
    if not logic_root.exists() or not nl_root.exists():
        return rows
    for task, pattern in COT_BARE_SAMPLE_PATTERNS.items():
        logic_rows = _load_sample_file(logic_root, pattern)
        nl_rows = _load_sample_file(nl_root, pattern)
        common = sorted(set(logic_rows) & set(nl_rows))
        if not common:
            continue
        selected = common[0]
        for doc_id in common:
            logic_score = _sample_metric(logic_rows[doc_id], task)
            nl_score = _sample_metric(nl_rows[doc_id], task)
            if logic_score > 0.0 or nl_score > 0.0:
                selected = doc_id
                break
        logic = logic_rows[selected]
        nl = nl_rows[selected]
        logic_raw = _raw_sample_response(logic)
        nl_raw = _raw_sample_response(nl)
        rows.append(
            {
                "task": task,
                "doc_id": selected,
                "question": _clip_for_report(_sample_question(logic), 800),
                "gold": _sample_gold(logic),
                "logic_metric": _sample_metric(logic, task),
                "logic_extracted": _extract_explicit_answer(logic_raw),
                "logic_generation": _clip_for_report(logic_raw),
                "nl_metric": _sample_metric(nl, task),
                "nl_extracted": _extract_explicit_answer(nl_raw),
                "nl_generation": _clip_for_report(nl_raw),
            }
        )
    if rows:
        write_csv(TABLE_DIR / "cot_bare_ood_generation_examples.csv", rows)
        lines = ["# Bare-Format OOD Generation Examples", ""]
        for row in rows:
            lines.extend(
                [
                    f"## {row['task']} doc_id {row['doc_id']}",
                    "",
                    f"Question: {row['question']}",
                    "",
                    f"Gold: {row['gold']}",
                    "",
                    f"Logic extracted: `{row['logic_extracted']}`; metric={float(row['logic_metric']):.3f}",
                    "",
                    "```text",
                    str(row["logic_generation"]),
                    "```",
                    "",
                    f"NL extracted: `{row['nl_extracted']}`; metric={float(row['nl_metric']):.3f}",
                    "",
                    "```text",
                    str(row["nl_generation"]),
                    "```",
                    "",
                ]
            )
        (OUT_ROOT / "ood_cot_bare_generation_examples_olmo7b_train1to25_seed3407.md").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )
    return rows


def latex_ood_examples(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No bare-format OOD sample rows were available at report generation time."

    def esc(text: object) -> str:
        return str(text).replace("_", r"\_")

    parts: list[str] = []
    for row in rows:
        metric_label = "EM" if row["task"] == "gsm8k" else "F1"
        parts.append(
            rf"""\subsection*{{{row['task']} doc {row['doc_id']}}}
\textbf{{Question.}} {esc(row['question'])}

\textbf{{Gold.}} {esc(row['gold'])}

\textbf{{Logic extracted:}} {esc(row['logic_extracted'])} ({metric_label}={float(row['logic_metric']):.3f})
\begin{{verbatim}}
{row['logic_generation']}
\end{{verbatim}}
\textbf{{NL extracted:}} {esc(row['nl_extracted'])} ({metric_label}={float(row['nl_metric']):.3f})
\begin{{verbatim}}
{row['nl_generation']}
\end{{verbatim}}"""
        )
    return "\n\n".join(parts)


SUPPLEMENTAL_FIGURE_CAPTIONS = {
    "ablation_same_target_token_budget_vs_main.pdf": (
        "Same target-token budget ablation compared with the main train-1-to-25 logic and NL baselines."
    ),
    "ablation_symbol_padded_depth_curve_train1to25.pdf": (
        "Symbol-padded equal-length logic depth curve for train-1-to-25. This older length-control view is kept "
        "alongside the cleaner compact/wordified comparison."
    ),
    "olmo7b_checkpoint_correct_k8_k16.pdf": (
        "Earlier combined OLMo-7B checkpoint curves for correct@8 and correct@16. The main text uses separated "
        "logic/NL @16 panels, but this aggregate view is retained for completeness."
    ),
    "olmo7b_checkpoint_joint_k8_k16.pdf": (
        "Earlier combined OLMo-7B checkpoint curves for joint correct+valid at @8 and @16. The main text uses "
        "separated logic/NL @16 panels."
    ),
    "qwen7b_partial_ood_correct_joint.pdf": (
        "Initial Qwen-2.5-7B partial architecture-ablation view. Superseded by the full architecture tables and "
        "figures, but retained as a generated result artifact."
    ),
    "sample_generation_panels.pdf": (
        "Synthetic sample-generation panels with extracted answers, gold answers, correctness, and validity metadata."
    ),
    "tiny_llama_100m_depth_correct_joint.pdf": (
        "Earlier combined Tiny Llama 100M depth curve showing correct and joint metrics together."
    ),
    "tiny_llama_200m_depth_correct_joint.pdf": (
        "Earlier combined Tiny Llama 200M depth curve showing correct and joint metrics together."
    ),
    "tiny_llama_50m_depth_correct_joint.pdf": (
        "Earlier combined Tiny Llama 50M depth curve showing correct and joint metrics together."
    ),
    "tiny_llama_checkpoint_correct_k8.pdf": (
        "Earlier all-size Tiny Llama checkpoint curve for correct@8. Size-separated checkpoint plots are in the main text."
    ),
    "tiny_llama_checkpoint_joint_k8.pdf": (
        "Earlier all-size Tiny Llama checkpoint curve for joint correct+valid@8. Size-separated checkpoint plots are in the main text."
    ),
    "tiny_llama_final_bands_correct_joint.pdf": (
        "Earlier aggregate Tiny Llama final-band view combining correct and joint metrics."
    ),
}


def default_figure_caption(name: str) -> str:
    stem = Path(name).stem.replace("_", " ")
    return f"Generated supplemental figure: {stem}."


def build_supplemental_figures_block(tex: str) -> str:
    included = set(re.findall(r"\{figures/([^}]+\.pdf)\}", tex))
    missing = [path.name for path in sorted(FIG_DIR.glob("*.pdf")) if path.name not in included]
    if not missing:
        return ""

    parts = [
        r"\clearpage",
        r"\section{Supplemental generated figures}",
        (
            "This section embeds every remaining generated PDF figure that was not already placed in the main "
            "narrative above. Some panels are older aggregate views now superseded by clearer separated plots; "
            "they are retained here so the in-repo report contains the full current figure set."
        ),
    ]
    for name in missing:
        caption = SUPPLEMENTAL_FIGURE_CAPTIONS.get(name, default_figure_caption(name))
        parts.append(
            rf"""\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/{name}}}
\caption{{{caption}}}
\end{{figure}}"""
        )
    return "\n\n".join(parts)


def build_artifact_index_block() -> str:
    csv_names = [path.name for path in sorted(TABLE_DIR.glob("*.csv"))]
    sample_names = [
        path.name
        for path in sorted(OUT_ROOT.glob("*.md"))
        if path.name.endswith(".md")
    ]
    pdf_names = [path.name for path in sorted(FIG_DIR.glob("*.pdf"))]

    def verbatim_block(title: str, names: list[str]) -> str:
        if not names:
            return f"\\paragraph{{{title}.}} None generated."
        return "\n".join(
            [
                rf"\paragraph{{{title}.}}",
                r"\begin{verbatim}",
                "\n".join(names),
                r"\end{verbatim}",
            ]
        )

    return "\n\n".join(
        [
            "The following generated artifacts are part of this report bundle.",
            verbatim_block("CSV result tables", csv_names),
            verbatim_block("PDF figures", pdf_names),
            verbatim_block("Markdown sample supplements", sample_names),
        ]
    )


def main_checkpoint_note(records: list[Record]) -> str:
    if not records:
        return "No main OLMo checkpoint pass@k files were present at report generation time."
    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for record in records:
        if record.checkpoint is not None:
            grouped[(record.template, record.train_max, record.seed)].append(record.checkpoint)

    expected_dense = list(range(1000, 10001, 1000))
    original_sparse = [1000, 3000, 10000]

    def label(item: tuple[tuple[str, int, int], list[int]]) -> str:
        (template, train_max, seed), _ = item
        template_label = "NL exact" if template == "nl_exact" else "logic"
        return f"{template_label} train1to{train_max} seed{seed}"

    full = [label(item) for item in grouped.items() if sorted(set(item[1])) == expected_dense]
    partial = [
        label(item)
        for item in grouped.items()
        if sorted(set(item[1])) != expected_dense and sorted(set(item[1])) != original_sparse
    ]
    sparse_count = sum(1 for steps in grouped.values() if sorted(set(steps)) == original_sparse)
    parts = [f"Main OLMo checkpoint pass@k files available: {len(records)}."]
    if full:
        parts.append("Full 1k-grid rows now available: " + ", ".join(full) + ".")
    if partial:
        parts.append("Rows with partial dense additions beyond 1k/3k/10k: " + ", ".join(partial) + ".")
    if sparse_count:
        parts.append(f"Rows still at the original 1k/3k/10k grid: {sparse_count}.")
    return " ".join(parts)


def build_corrected_branchproof_report_block() -> tuple[str, str, bool]:
    manifest_path = CORRECTED_BRANCHPROOF_ROOT / "manifest.json"
    qualitative_path = CORRECTED_BRANCHPROOF_ROOT / "qualitative_grid_audit.json"
    summary_path = CORRECTED_BRANCHPROOF_ROOT / "tables" / "final_group_summary.csv"
    required = (manifest_path, qualitative_path, summary_path)
    if not all(path.is_file() for path in required):
        return (
            "Corrected three-seed BranchProof aggregation is still pending its complete "
            "artifact and qualitative gate.",
            "Corrected BranchProof is still gated; no replacement quantitative claim is included.",
            False,
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualitative = json.loads(qualitative_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    ready = (
        manifest.get("final_json_count") == 30
        and not manifest.get("problems")
        and qualitative.get("accepted") is True
        and qualitative.get("observed_grid_size") == 30
        and len(summary) == 10
        and set(summary["n"].astype(int)) == {3}
    )
    if not ready:
        return (
            "Corrected BranchProof artifacts exist, but the complete 30-run structural and "
            "qualitative acceptance gate has not passed.",
            "Corrected BranchProof is still gated; no replacement quantitative claim is included.",
            False,
        )

    copy_pairs = (
        (
            CORRECTED_BRANCHPROOF_ROOT / "figures" / "final_primary_ood_correctness.pdf",
            FIG_DIR / "corrected_branchproof_primary_ood_correctness.pdf",
        ),
        (
            CORRECTED_BRANCHPROOF_ROOT / "figures" / "final_train25_sampling_efficiency.pdf",
            FIG_DIR / "corrected_branchproof_train25_sampling_efficiency.pdf",
        ),
        (summary_path, TABLE_DIR / "corrected_branchproof_final_group_summary.csv"),
        (
            CORRECTED_BRANCHPROOF_ROOT / "tables" / "paired_delta_summary.csv",
            TABLE_DIR / "corrected_branchproof_paired_delta_summary.csv",
        ),
    )
    for source, destination in copy_pairs:
        if source.is_file():
            shutil.copy2(source, destination)

    summary = summary.sort_values(["train_max", "template"])

    def percent_stat(row: pd.Series, metric: str) -> str:
        return (
            f"{100.0 * float(row[f'{metric}_mean']):.1f} $\\pm$ "
            f"{100.0 * float(row[f'{metric}_std']):.1f}"
        )

    table_lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Train & Trace & Greedy OOD & OOD c@1 & OOD joint@1 & OOD c@16 & OOD joint@16 \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        trace = "Logic" if row["template"] == "logic" else "Natural"
        table_lines.append(
            f"1--{int(row['train_max'])} & {trace} & "
            f"{percent_stat(row, 'greedy_ood_correct')} & "
            f"{percent_stat(row, 'ood_correct1')} & "
            f"{percent_stat(row, 'ood_joint1')} & "
            f"{percent_stat(row, 'ood_correct16')} & "
            f"{percent_stat(row, 'ood_joint16')} \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])

    train25 = summary[summary["train_max"] == 25].set_index("template")
    logic = train25.loc["logic"]
    natural = train25.loc["nl_exact"]
    verifier_block = ""
    verifier_path = CORRECTED_BRANCHPROOF_ROOT / "verifier_selection_train25.json"
    verifier_audits = [
        VERIFIER_BRANCHPROOF_AUDIT_ROOT / f"{modality}_seed{seed}.json"
        for modality in ("logic", "nl_exact")
        for seed in (3407, 3408, 3409)
    ]
    verifier_passk = [
        VERIFIER_BRANCHPROOF_PASSK_ROOT
        / f"sft_branchproof_unique_v2_{modality}_train1to25_10k_seed{seed}_verifier_fullsamples_passk.json"
        for modality in ("logic", "nl_exact")
        for seed in (3407, 3408, 3409)
    ]
    verifier_ready = (
        verifier_path.is_file()
        and all(path.is_file() for path in verifier_audits)
        and all(path.is_file() for path in verifier_passk)
    )
    if verifier_ready:
        verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
        verifier_ready = (
            verifier.get("selection_uses_gold_answer") is False
            and verifier.get("expected_k") == 16
            and len(verifier.get("files", [])) == 6
            and all(row.get("prompt_groups") == 448 for row in verifier.get("files", []))
            and all(
                json.loads(path.read_text(encoding="utf-8")).get("accepted") is True
                for path in verifier_audits
            )
        )
    if verifier_ready:
        sampling_rows: list[dict[str, object]] = []
        selection_rows: list[dict[str, object]] = []
        for modality, joint_metric in (
            ("logic", "citation_free_joint_pass"),
            ("nl_exact", "nl_logic_joint_pass"),
        ):
            seed_metrics = []
            for seed in (3407, 3408, 3409):
                path = VERIFIER_BRANCHPROOF_PASSK_ROOT / (
                    f"sft_branchproof_unique_v2_{modality}_train1to25_10k_seed{seed}_"
                    "verifier_fullsamples_passk.json"
                )
                seed_metrics.append(json.loads(path.read_text(encoding="utf-8"))["metrics"])
            for band, metric_band in (("OOD", "band_ood"), ("Depth 50", "step_50")):
                for k in (1, 2, 4, 8, 16):
                    correct = [
                        metrics[f"synthetic_sampled/{metric_band}/correct_pass@{k}"]
                        for metrics in seed_metrics
                    ]
                    joint = [
                        metrics[f"synthetic_sampled/{metric_band}/{joint_metric}@{k}"]
                        for metrics in seed_metrics
                    ]
                    sampling_rows.append(
                        {
                            "trace": modality,
                            "band": band,
                            "k": k,
                            "answer_mean": mean(correct),
                            "answer_std": pstdev(correct),
                            "joint_mean": mean(joint),
                            "joint_std": pstdev(joint),
                        }
                    )
            selected = verifier["summary"][modality]["ood"]
            for strategy, prefix in (
                ("First valid", "first_valid"),
                ("Max line-valid", "max_line"),
                ("Oracle ceiling", "oracle"),
            ):
                selection_rows.append(
                    {
                        "trace": modality,
                        "strategy": strategy,
                        "answer_mean": selected[f"{prefix}_correct"]["mean"],
                        "answer_std": selected[f"{prefix}_correct"]["std"],
                        "joint_mean": selected[f"{prefix}_joint"]["mean"],
                        "joint_std": selected[f"{prefix}_joint"]["std"],
                    }
                )
        write_csv(TABLE_DIR / "corrected_branchproof_verifier_sampling_curve.csv", sampling_rows)
        write_csv(TABLE_DIR / "corrected_branchproof_verifier_selection.csv", selection_rows)

        sampling_table = [
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Trace & $k$ & OOD answer & OOD joint & Depth-50 joint \\",
            r"\midrule",
        ]
        for modality in ("logic", "nl_exact"):
            trace = "Logic" if modality == "logic" else "Natural"
            for k in (1, 4, 8, 16):
                ood = next(
                    row for row in sampling_rows
                    if row["trace"] == modality and row["band"] == "OOD" and row["k"] == k
                )
                depth50 = next(
                    row for row in sampling_rows
                    if row["trace"] == modality and row["band"] == "Depth 50" and row["k"] == k
                )
                sampling_table.append(
                    f"{trace} & {k} & {100 * float(ood['answer_mean']):.1f} $\\pm$ "
                    f"{100 * float(ood['answer_std']):.1f} & "
                    f"{100 * float(ood['joint_mean']):.1f} $\\pm$ "
                    f"{100 * float(ood['joint_std']):.1f} & "
                    f"{100 * float(depth50['joint_mean']):.1f} $\\pm$ "
                    f"{100 * float(depth50['joint_std']):.1f} \\\\"
                )
        sampling_table.extend([r"\bottomrule", r"\end{tabular}"])

        selection_table = [
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Trace & Selection from 16 & OOD answer & OOD joint \\",
            r"\midrule",
        ]
        for row in selection_rows:
            trace = "Logic" if row["trace"] == "logic" else "Natural"
            selection_table.append(
                f"{trace} & {row['strategy']} & "
                f"{100 * float(row['answer_mean']):.1f} $\\pm$ "
                f"{100 * float(row['answer_std']):.1f} & "
                f"{100 * float(row['joint_mean']):.1f} $\\pm$ "
                f"{100 * float(row['joint_std']):.1f} \\\\"
            )
        selection_table.extend([r"\bottomrule", r"\end{tabular}"])
        verifier_block = rf"""
\subsection{{Verifier-guided selection from retained generations}}
All six train-1-to-25 full-retention rows contain 448 prompt groups and 16 generations per
prompt. The fail-closed audit accepted all 43,008 generations, including fresh-constant and
credited-validity checks. The sampling curve remains strongly separated before selection.

\begin{{table}}[H]
\centering
\scriptsize
{chr(10).join(sampling_table)}
\caption{{Corrected BranchProof sampling efficiency over the fully retained generations.
Joint is modality-appropriate citation-free correct-and-valid performance.}}
\end{{table}}

The first-valid and maximum-line-valid selectors use only verifier diagnostics and never the
gold answer. The oracle row is reported only as a coverage ceiling.

\begin{{table}}[H]
\centering
\scriptsize
{chr(10).join(selection_table)}
\caption{{OOD selection from 16 retained candidates. First-valid and max-line-valid are
non-oracle; the oracle ceiling uses the gold answer and is not a deployable selector.}}
\end{{table}}

For logic, first-valid selection reaches
{100 * verifier['summary']['logic']['ood']['first_valid_correct']['mean']:.1f}$\pm${100 * verifier['summary']['logic']['ood']['first_valid_correct']['std']:.1f}
percent OOD answer accuracy and
{100 * verifier['summary']['logic']['ood']['first_valid_joint']['mean']:.1f}$\pm${100 * verifier['summary']['logic']['ood']['first_valid_joint']['std']:.1f}
percent joint accuracy. The corresponding natural-language values are
{100 * verifier['summary']['nl_exact']['ood']['first_valid_correct']['mean']:.1f}$\pm${100 * verifier['summary']['nl_exact']['ood']['first_valid_correct']['std']:.1f}
and {100 * verifier['summary']['nl_exact']['ood']['first_valid_joint']['mean']:.1f}$\pm${100 * verifier['summary']['nl_exact']['ood']['first_valid_joint']['std']:.1f} percent.
Raw depth-50 review shows complete correct-valid chains and genuine answer-correct-invalid
formal traces. Natural-language failures predominantly copy premises until truncation: across
seeds, 27--32 percent omit a closing answer, versus 1.1--2.8 percent for logic. Thus the
selection result reflects both proof checkability and a large representation-dependent
long-context generation failure; it is not evidence that a verifier alone closes the gap.
"""
    executive = (
        "The corrected 30-run BranchProof grid passed all row and cross-grid gates. At "
        "train-1-to-25, logic leads natural-language supervision on OOD greedy correctness "
        f"({100 * logic['greedy_ood_correct_mean']:.1f} versus "
        f"{100 * natural['greedy_ood_correct_mean']:.1f}) and sampled pass@1 correctness "
        f"({100 * logic['ood_correct1_mean']:.1f} versus "
        f"{100 * natural['ood_correct1_mean']:.1f}), with a similarly large joint-validity gap. "
        "The reversal is specific to the deepest training range: natural language is stronger "
        "at greedy/pass@1 for train maxima 5--20, so the result supports a depth-dependent "
        "formal advantage rather than uniform superiority."
    )
    table = "\n".join(table_lines)
    block = rf"""
\section{{Corrected BranchProof result}}
The replacement grid contains 30 three-seed runs and passed every row-level artifact gate plus
the cross-grid qualitative audit. Every retained prompt uses fresh constants through its requested
depth, and no citation-free-valid retained sample carries a validation error or a line-valid
fraction below one. The table reports percentage mean $\pm$ population standard deviation across
three seeds. OOD means evaluation depths above the run's maximum training depth.

\begin{{table}}[H]
\centering
\scriptsize
{table}
\caption{{Corrected unique-answer BranchProof primary results. Joint validity is citation-free
formal validity for Logic and translated citation-free validity for Natural.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\linewidth]{{figures/corrected_branchproof_primary_ood_correctness.pdf}}
\caption{{Greedy and sampled pass@1 OOD correctness. Natural-language supervision is stronger
for train maxima 5--20, but logic reverses the comparison at train-1-to-25.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\linewidth]{{figures/corrected_branchproof_train25_sampling_efficiency.pdf}}
\caption{{Train-1-to-25 OOD correctness and modality-appropriate correct-and-valid pass@k.
The formal advantage is already present at pass@1 and is not only a large-sample-budget effect.}}
\end{{figure}}

At train-1-to-25, formal supervision reaches greedy OOD correctness
{100 * logic['greedy_ood_correct_mean']:.1f}$\pm${100 * logic['greedy_ood_correct_std']:.1f}
versus {100 * natural['greedy_ood_correct_mean']:.1f}$\pm${100 * natural['greedy_ood_correct_std']:.1f}
for natural supervision. Sampled OOD pass@1 correctness is
{100 * logic['ood_correct1_mean']:.1f}$\pm${100 * logic['ood_correct1_std']:.1f} versus
{100 * natural['ood_correct1_mean']:.1f}$\pm${100 * natural['ood_correct1_std']:.1f}; the corresponding
correct-and-valid values are {100 * logic['ood_joint1_mean']:.1f}$\pm${100 * logic['ood_joint1_std']:.1f}
and {100 * natural['ood_joint1_mean']:.1f}$\pm${100 * natural['ood_joint1_std']:.1f}.
Representative failures explain why correctness and validity remain separate: long formal
generations can reach the right answer through an invalid branch, while many natural-language
failures copy premises until the shared cap and never emit an answer. Selected length, shortcut,
hybrid, conditioned-dual, and architecture controls that passed the same gates are reported next.

{verifier_block}
"""
    return block, executive, True


def build_selected_branchproof_report_block() -> tuple[str, str, bool]:
    families = {
        "Symbol-padded formal": {
            "subdir": "surface",
            "stem": "sft_branchproof_unique_v2_surface_logic_symbol_padded_train1to25_10k_seed{seed}",
            "audits": ("surface_0.json", "surface_1.json", "surface_2.json"),
        },
        "Terse natural": {
            "subdir": "surface",
            "stem": "sft_branchproof_unique_v2_surface_terse_nl_train1to25_10k_seed{seed}",
            "audits": ("surface_6.json", "surface_7.json", "surface_8.json"),
            "joint_metric": "nl_logic_joint_pass",
        },
        "Target-token-matched natural": {
            "subdir": "surface",
            "stem": (
                "sft_branchproof_unique_v2_same_target_tokens_nl_exact_"
                "train1to25_7140steps_seed{seed}"
            ),
            "audits": ("surface_24.json", "surface_25.json", "surface_26.json"),
            "joint_metric": "nl_logic_joint_pass",
        },
        "Shortcut-trained formal": {
            "subdir": "shortcut",
            "stem": (
                "sft_branchproof_unique_v2_shortcut_schema_0p8_logic_"
                "train1to25_10k_seed{seed}"
            ),
            "audits": ("shortcut_12.json", "shortcut_13.json", "shortcut_14.json"),
        },
        "Shortcut-trained natural": {
            "subdir": "shortcut",
            "stem": (
                "sft_branchproof_unique_v2_shortcut_schema_0p8_nl_exact_"
                "train1to25_10k_seed{seed}"
            ),
            "audits": ("shortcut_15.json", "shortcut_16.json", "shortcut_17.json"),
            "joint_metric": "nl_logic_joint_pass",
        },
        "NL-then-formal hybrid": {
            "subdir": "hybrid",
            "stem": "sft_branchproof_unique_v2_hybrid_think_formal_train1to25_10k_seed{seed}",
            "audits": ("hybrid_12.json", "hybrid_13.json", "hybrid_14.json"),
        },
        "Formal-then-NL hybrid": {
            "subdir": "hybrid",
            "stem": "sft_branchproof_unique_v2_hybrid_formal_think_train1to25_10k_seed{seed}",
            "audits": ("hybrid_27.json", "hybrid_28.json", "hybrid_29.json"),
        },
        "Conditioned-dual formal": {
            "subdir": "conditioned10k",
            "stem": (
                "sft_branchproof_unique_v2_conditioned10k_train1to25_10k_"
                "seed{seed}_conditioned_logic"
            ),
            "audits": ("conditioned10k_24.json", "conditioned10k_26.json", "conditioned10k_28.json"),
        },
        "Conditioned-dual NL": {
            "subdir": "conditioned10k",
            "stem": (
                "sft_branchproof_unique_v2_conditioned10k_train1to25_10k_"
                "seed{seed}_conditioned_nl"
            ),
            "audits": ("conditioned10k_25.json", "conditioned10k_27.json", "conditioned10k_29.json"),
            "joint_metric": "nl_logic_joint_pass",
        },
        "Qwen2.5-7B formal": {
            "subdir": "architecture",
            "stem": (
                "sft_branchproof_unique_v2_arch_qwen2p5_7b_logic_"
                "train1to25_10k_seed{seed}"
            ),
            "audits": ("architecture_24.json", "architecture_25.json", "architecture_26.json"),
        },
        "Qwen2.5-7B natural": {
            "subdir": "architecture",
            "stem": (
                "sft_branchproof_unique_v2_arch_qwen2p5_7b_nl_exact_"
                "train1to25_10k_seed{seed}"
            ),
            "audits": ("architecture_33.json", "architecture_34.json", "architecture_35.json"),
            "joint_metric": "nl_logic_joint_pass",
        },
        "OLMo-3-32B formal": {
            "subdir": "large",
            "stem": (
                "sft_branchproof_unique_v2_arch_olmo3_1125_32b_logic_"
                "train1to25_10k_seed{seed}"
            ),
            "audits": ("large_0.json", "large_1.json", "large_2.json"),
        },
        "OLMo-3-32B natural": {
            "subdir": "large",
            "stem": (
                "sft_branchproof_unique_v2_arch_olmo3_1125_32b_nl_exact_"
                "train1to25_10k_seed{seed}"
            ),
            "audits": ("large_3.json", "large_4.json", "large_5.json"),
            "joint_metric": "nl_logic_joint_pass",
        },
        "OLMo-3-32B conditioned formal": {
            "subdir": "large",
            "stem": (
                "sft_branchproof_unique_v2_arch_olmo3_1125_32b_conditioned_dual_"
                "train1to25_10k_seed{seed}_conditioned_logic"
            ),
            "audits": ("large_12.json", "large_14.json", "large_16.json"),
        },
        "OLMo-3-32B conditioned natural": {
            "subdir": "large",
            "stem": (
                "sft_branchproof_unique_v2_arch_olmo3_1125_32b_conditioned_dual_"
                "train1to25_10k_seed{seed}_conditioned_nl"
            ),
            "audits": ("large_13.json", "large_15.json", "large_17.json"),
            "joint_metric": "nl_logic_joint_pass",
        },
    }
    for spec in families.values():
        audit_paths = [SELECTED_BRANCHPROOF_AUDIT_ROOT / name for name in spec["audits"]]
        if not all(path.is_file() for path in audit_paths):
            return "", "", False
        if not all(json.loads(path.read_text(encoding="utf-8")).get("accepted") is True for path in audit_paths):
            return "", "", False

    def summarize_family(label: str, spec: dict[str, object]) -> dict[str, object]:
        seed_rows: list[dict[str, float]] = []
        for seed in (3407, 3408, 3409):
            path = (
                SELECTED_BRANCHPROOF_PASSK_ROOT
                / str(spec["subdir"])
                / f"{str(spec['stem']).format(seed=seed)}_passk.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            depths = (30, 35, 40, 45, 50)
            joint_metric = str(spec.get("joint_metric", "citation_free_joint_pass"))
            seed_rows.append(
                {
                    "greedy_ood_correct": mean(
                        float(metrics[f"synthetic/step_{depth}/correct"]) for depth in depths
                    ),
                    "ood_correct1": mean(
                        float(metrics[f"synthetic_sampled/step_{depth}/correct_pass@1"])
                        for depth in depths
                    ),
                    "ood_joint1": mean(
                        float(metrics[f"synthetic_sampled/step_{depth}/{joint_metric}@1"])
                        for depth in depths
                    ),
                    "ood_correct16": mean(
                        float(metrics[f"synthetic_sampled/step_{depth}/correct_pass@16"])
                        for depth in depths
                    ),
                    "ood_joint16": mean(
                        float(metrics[f"synthetic_sampled/step_{depth}/{joint_metric}@16"])
                        for depth in depths
                    ),
                    "depth50_correct1": float(
                        metrics["synthetic_sampled/step_50/correct_pass@1"]
                    ),
                    "depth50_joint1": float(
                        metrics[f"synthetic_sampled/step_50/{joint_metric}@1"]
                    ),
                }
            )
        row: dict[str, object] = {"condition": label, "n": len(seed_rows)}
        for metric in seed_rows[0]:
            values = [seed_row[metric] for seed_row in seed_rows]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values)
        return row

    rows = [summarize_family(label, spec) for label, spec in families.items()]
    write_csv(TABLE_DIR / "corrected_branchproof_selected_controls.csv", rows)

    def percent_stat(row: dict[str, object], metric: str) -> str:
        return (
            f"{100.0 * float(row[f'{metric}_mean']):.1f} $\\pm$ "
            f"{100.0 * float(row[f'{metric}_std']):.1f}"
        )

    table_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & Greedy OOD & OOD c@1 & OOD joint@1 & OOD c@16 & OOD joint@16 \\",
        r"\midrule",
    ]
    for row in rows:
        table_lines.append(
            f"{row['condition']} & "
            f"{percent_stat(row, 'greedy_ood_correct')} & "
            f"{percent_stat(row, 'ood_correct1')} & "
            f"{percent_stat(row, 'ood_joint1')} & "
            f"{percent_stat(row, 'ood_correct16')} & "
            f"{percent_stat(row, 'ood_joint16')} \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])

    by_condition = {str(row["condition"]): row for row in rows}
    surface = by_condition["Symbol-padded formal"]
    terse_nl = by_condition["Terse natural"]
    target_nl = by_condition["Target-token-matched natural"]
    shortcut_logic = by_condition["Shortcut-trained formal"]
    shortcut_nl = by_condition["Shortcut-trained natural"]
    hybrid = by_condition["NL-then-formal hybrid"]
    reverse_hybrid = by_condition["Formal-then-NL hybrid"]
    conditioned_logic = by_condition["Conditioned-dual formal"]
    conditioned_nl = by_condition["Conditioned-dual NL"]
    qwen_logic = by_condition["Qwen2.5-7B formal"]
    qwen_nl = by_condition["Qwen2.5-7B natural"]
    olmo32_logic = by_condition["OLMo-3-32B formal"]
    olmo32_nl = by_condition["OLMo-3-32B natural"]
    olmo32_conditioned_logic = by_condition["OLMo-3-32B conditioned formal"]
    olmo32_conditioned_nl = by_condition["OLMo-3-32B conditioned natural"]
    executive = (
        "Fifteen corrected train-1-to-25 control conditions now pass complete three-seed row "
        "and raw-generation gates. Symbol padding preserves most of the formal OOD advantage "
        f"({100 * float(surface['ood_correct1_mean']):.1f}% answer pass@1), while the "
        "two same-example hybrid orders collapse under extrapolation "
        f"({100 * float(hybrid['ood_correct1_mean']):.1f}% and "
        f"{100 * float(reverse_hybrid['ood_correct1_mean']):.1f}% answer pass@1). "
        f"Shortcut-trained formal versus natural reaches "
        f"{100 * float(shortcut_logic['ood_correct1_mean']):.1f}% versus "
        f"{100 * float(shortcut_nl['ood_correct1_mean']):.1f}%, and Qwen2.5-7B reaches "
        f"{100 * float(qwen_logic['ood_correct1_mean']):.1f}% versus "
        f"{100 * float(qwen_nl['ood_correct1_mean']):.1f}%. In the conditioned-dual "
        f"checkpoint, formal prompting retains {100 * float(conditioned_logic['ood_correct1_mean']):.1f}% "
        f"OOD answer pass@1 versus {100 * float(conditioned_nl['ood_correct1_mean']):.1f}% for NL. "
        f"At OLMo-3-32B, formal versus natural reaches "
        f"{100 * float(olmo32_logic['ood_correct1_mean']):.1f}% versus "
        f"{100 * float(olmo32_nl['ood_correct1_mean']):.1f}%; the shared conditioned checkpoint "
        f"reaches {100 * float(olmo32_conditioned_logic['ood_correct1_mean']):.1f}% in formal mode "
        f"and {100 * float(olmo32_conditioned_nl['ood_correct1_mean']):.1f}% in natural mode."
    )
    block = rf"""
\subsection{{Accepted corrected BranchProof controls}}
The selected syntax/length, shortcut, hybrid-order, conditioned-dual, Qwen2.5-7B, and
single-modal and conditioned OLMo-3-32B controls are complete across three seeds. All accepted rows
pass the same 448-prompt, 16-generation, 14-depth artifact and qualitative gates as the corrected
main grid. OOD contains depths 30--50. Joint uses citation-free formal validity for formal
outputs and translated citation-free validity for natural-language outputs.

\begin{{table}}[H]
\centering
\scriptsize
{chr(10).join(table_lines)}
\caption{{Accepted corrected train-1-to-25 controls, percentage mean $\pm$ population standard
deviation over three seeds.}}
\end{{table}}

Symbol padding leaves the formal result direction intact, although its OOD pass@1 correctness
({100 * float(surface['ood_correct1_mean']):.1f}$\pm${100 * float(surface['ood_correct1_std']):.1f})
is below compact formal supervision. The NL-then-formal hybrid is perfect at the training edge in
the inspected samples but collapses beyond it: long generations copy the NL premises and proof
before starting or completing the formal trace, then truncate without an answer. Its OOD answer
pass@16 is only {100 * float(hybrid['ood_correct16_mean']):.1f}$\pm$
{100 * float(hybrid['ood_correct16_std']):.1f}; reversing the order also collapses, with OOD
answer pass@1 {100 * float(reverse_hybrid['ood_correct1_mean']):.1f}$\pm$
{100 * float(reverse_hybrid['ood_correct1_std']):.1f}. Terse natural supervision reaches
{100 * float(terse_nl['ood_correct1_mean']):.1f}$\pm$
{100 * float(terse_nl['ood_correct1_std']):.1f} OOD answer pass@1, while matching the natural
target-token exposure reaches {100 * float(target_nl['ood_correct1_mean']):.1f}$\pm$
{100 * float(target_nl['ood_correct1_std']):.1f} and has substantial seed variance.
Shortcut-trained formal versus natural reaches
{100 * float(shortcut_logic['ood_correct1_mean']):.1f}$\pm$
{100 * float(shortcut_logic['ood_correct1_std']):.1f} versus
{100 * float(shortcut_nl['ood_correct1_mean']):.1f}$\pm$
{100 * float(shortcut_nl['ood_correct1_std']):.1f}; Qwen2.5-7B reaches
{100 * float(qwen_logic['ood_correct1_mean']):.1f}$\pm$
{100 * float(qwen_logic['ood_correct1_std']):.1f} versus
{100 * float(qwen_nl['ood_correct1_mean']):.1f}$\pm$
{100 * float(qwen_nl['ood_correct1_std']):.1f}. The conditioned-dual checkpoint also favors formal
prompting: OOD answer/joint pass@1 is
{100 * float(conditioned_logic['ood_correct1_mean']):.1f}$\pm$
{100 * float(conditioned_logic['ood_correct1_std']):.1f} /
{100 * float(conditioned_logic['ood_joint1_mean']):.1f}$\pm$
{100 * float(conditioned_logic['ood_joint1_std']):.1f}, versus
{100 * float(conditioned_nl['ood_correct1_mean']):.1f}$\pm$
{100 * float(conditioned_nl['ood_correct1_std']):.1f} /
{100 * float(conditioned_nl['ood_joint1_mean']):.1f}$\pm$
{100 * float(conditioned_nl['ood_joint1_std']):.1f} for NL. Retained conditioned-NL samples are
clean through roughly depth 30, then copy long premise/proof prefixes and usually omit the answer.
The matched OLMo-3-32B single-modal comparison reaches OOD answer/joint pass@1
{100 * float(olmo32_logic['ood_correct1_mean']):.1f}$\pm$
{100 * float(olmo32_logic['ood_correct1_std']):.1f} /
{100 * float(olmo32_logic['ood_joint1_mean']):.1f}$\pm$
{100 * float(olmo32_logic['ood_joint1_std']):.1f} for formal versus
{100 * float(olmo32_nl['ood_correct1_mean']):.1f}$\pm$
{100 * float(olmo32_nl['ood_correct1_std']):.1f} /
{100 * float(olmo32_nl['ood_joint1_mean']):.1f}$\pm$
{100 * float(olmo32_nl['ood_joint1_std']):.1f} for natural supervision. The shared conditioned
OLMo-3-32B checkpoint reaches OOD answer/joint pass@1
{100 * float(olmo32_conditioned_logic['ood_correct1_mean']):.1f}$\pm$
{100 * float(olmo32_conditioned_logic['ood_correct1_std']):.1f} /
{100 * float(olmo32_conditioned_logic['ood_joint1_mean']):.1f}$\pm$
{100 * float(olmo32_conditioned_logic['ood_joint1_std']):.1f} under formal prompting and
{100 * float(olmo32_conditioned_nl['ood_correct1_mean']):.1f}$\pm$
{100 * float(olmo32_conditioned_nl['ood_correct1_std']):.1f} /
{100 * float(olmo32_conditioned_nl['ood_joint1_mean']):.1f}$\pm$
{100 * float(olmo32_conditioned_nl['ood_joint1_std']):.1f} under natural prompting. This is a
within-checkpoint output-mode comparison; the single-modal rows remain the matched independent
checkpoint controls.
"""
    return block, executive, True


def write_report(
    main_summary: pd.DataFrame,
    tiny_summary: pd.DataFrame,
    tiny_100k_summary: pd.DataFrame,
    qwen_summary: pd.DataFrame,
    architecture_summary: pd.DataFrame,
    main_ood_summary: pd.DataFrame,
    tiny_ood_summary: pd.DataFrame,
    main_cot_bare_ood_summary: pd.DataFrame,
    tiny_cot_bare_ood_summary: pd.DataFrame,
    tiny_100k_cot_bare_ood_summary: pd.DataFrame,
    olmo32_cot_bare_gsm8k: pd.DataFrame,
    ablation_summaries: dict[str, pd.DataFrame],
    paired_full_summary: pd.DataFrame,
    paired_igsm_semantic_summary: pd.DataFrame,
    active_paired_partial_summary: pd.DataFrame,
    batch_size_partial_summary: pd.DataFrame,
    cot_bare_examples: list[dict[str, object]],
    main_ckpt_note: str,
    tiny_ckpt_count: int,
    tiny_100k_ckpt_count: int,
) -> None:
    corrected_branchproof_block, corrected_branchproof_executive, corrected_branchproof_ready = (
        build_corrected_branchproof_report_block()
    )
    selected_branchproof_block, selected_branchproof_executive, selected_branchproof_ready = (
        build_selected_branchproof_report_block()
    )
    main_rows = main_summary[(main_summary["size"] == "") & (main_summary["n"] >= 1)].copy()
    main_rows = main_rows.sort_values(["template", "train_max"])
    main_ood_rows = main_ood_summary.sort_values(["template", "train_max"]) if not main_ood_summary.empty else pd.DataFrame()
    main_cot_bare_rows = (
        main_cot_bare_ood_summary.sort_values(["template", "train_max"])
        if not main_cot_bare_ood_summary.empty
        else pd.DataFrame()
    )
    tiny_rows = tiny_summary[tiny_summary["size"] != ""].copy() if not tiny_summary.empty else pd.DataFrame()
    if not tiny_rows.empty:
        tiny_rows["size_order"] = tiny_rows["size"].map(SIZE_ORDER)
        tiny_rows = tiny_rows.sort_values(["size_order", "template"])
    tiny_100k_rows = (
        tiny_100k_summary[tiny_100k_summary["size"] != ""].copy() if not tiny_100k_summary.empty else pd.DataFrame()
    )
    if not tiny_100k_rows.empty:
        tiny_100k_rows["size_order"] = tiny_100k_rows["size"].map(SIZE_ORDER)
        tiny_100k_rows = tiny_100k_rows.sort_values(["size_order", "template"])
    qwen_rows = qwen_summary.sort_values(["template", "train_max"]) if not qwen_summary.empty else pd.DataFrame()
    architecture_rows = architecture_summary.copy() if not architecture_summary.empty else pd.DataFrame()
    if not architecture_rows.empty:
        architecture_rows["model_order"] = architecture_rows["model"].map(MODEL_ORDER).fillna(99)
        architecture_rows = architecture_rows.sort_values(["model_order", "template", "train_max"]).drop(columns=["model_order"])
    token_length_rows = pd.DataFrame(
        [
            {"train_range": "1..5", "logic_target": 322, "nl_target": 382, "logic_total": 697, "nl_total": 757},
            {"train_range": "1..10", "logic_target": 500, "nl_target": 653, "logic_total": 1166, "nl_total": 1319},
            {"train_range": "1..15", "logic_target": 681, "nl_target": 925, "logic_total": 1637, "nl_total": 1881},
            {"train_range": "1..20", "logic_target": 863, "nl_target": 1196, "logic_total": 2109, "nl_total": 2442},
            {"train_range": "1..25", "logic_target": 1049, "nl_target": 1469, "logic_total": 2587, "nl_total": 3008},
        ]
    )
    token_budget_rows = ablation_summaries.get("token_budget", pd.DataFrame())
    token_budget_comparison_rows = build_token_budget_comparison_table(main_summary, token_budget_rows)
    token_budget_exposure_rows = build_token_budget_exposure_table()
    symbol_padded_token_rows = build_symbol_padded_token_match_table()
    ablation_example_rows = build_ablation_training_examples()
    ablation_token_rows = build_ablation_token_audit()
    length_control_surface_rows = build_length_control_surface_examples()
    symbol_padded_eval_rows = build_symbol_padded_eval_comparison_table(
        main_summary,
        ablation_summaries.get("symbol_padded", pd.DataFrame()),
        ablation_summaries.get("wordified", pd.DataFrame()),
    )
    shortcut_rows = ablation_summaries.get("shortcut", pd.DataFrame())
    shortcut_kind_rows = ablation_summaries.get("shortcut_kind", pd.DataFrame())
    trace_control_rows = ablation_summaries.get("trace_control", pd.DataFrame())
    hybrid_order_rows = ablation_summaries.get("hybrid_order", pd.DataFrame())
    conditioned_rows = ablation_summaries.get("conditioned", pd.DataFrame())
    conditioned_50k_rows = ablation_summaries.get("conditioned_50k", pd.DataFrame())
    conditioned_50k_eval_count = int(conditioned_50k_rows["n"].sum()) if not conditioned_50k_rows.empty else 0
    if conditioned_50k_eval_count >= 30:
        conditioned_50k_status_sentence = "The 50k final eval is complete at 30/30 rows."
    elif conditioned_50k_eval_count:
        conditioned_50k_status_sentence = (
            f"The 50k final eval is partial at {conditioned_50k_eval_count}/30 rows; "
            "the n column marks seed coverage and the missing conditioned-NL rows are in targeted recovery."
        )
    else:
        conditioned_50k_status_sentence = "The 50k final eval has not written pass@k rows yet."
    experiment_status_rows = build_experiment_artifact_status()
    paired_full_rows = paired_full_summary.copy() if not paired_full_summary.empty else pd.DataFrame()
    if not paired_full_rows.empty:
        paired_full_rows = paired_full_rows.sort_values(["family", "template", "train_max"])
    paired_igsm_semantic_rows = (
        paired_igsm_semantic_summary.copy() if not paired_igsm_semantic_summary.empty else pd.DataFrame()
    )
    if not paired_igsm_semantic_rows.empty:
        paired_igsm_semantic_rows = paired_igsm_semantic_rows.sort_values(["template", "train_max"])
    active_paired_partial_rows = (
        active_paired_partial_summary.copy() if not active_paired_partial_summary.empty else pd.DataFrame()
    )
    if not active_paired_partial_rows.empty:
        active_paired_partial_rows = active_paired_partial_rows.sort_values(["family", "template", "train_max"])
    batch_size_partial_rows = batch_size_partial_summary.copy() if not batch_size_partial_summary.empty else pd.DataFrame()
    if not batch_size_partial_rows.empty:
        batch_size_partial_rows = batch_size_partial_rows.sort_values(["eval_condition", "batch_size"])
        batch_size_partial_rows["batch_size_label"] = batch_size_partial_rows["batch_size"].astype(int).astype(str)
    batch_size_diagnostic_rows = build_hfsa_batch_size_diagnostic_table(batch_size_partial_summary)
    shortcut_comparison_rows = build_shortcut_comparison_table(main_summary, shortcut_rows)
    conditioned_comparison_rows = build_conditioned_comparison_table(main_summary, conditioned_rows)
    trace_control_comparison_rows = build_trace_control_with_baselines(main_summary, trace_control_rows)
    hybrid_order_comparison_rows = build_hybrid_order_with_baselines(main_summary, hybrid_order_rows)
    train_maxes = [5, 10, 15, 20, 25]
    tiny_sizes = ["50m", "100m", "200m"]
    trace_control_figure_block = ""
    if not trace_control_rows.empty and (FIG_DIR / "ablation_trace_controls_summary.pdf").exists():
        trace_control_figure_block = r"""
\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{figures/ablation_trace_controls_summary.pdf}
\caption{Trace-control ablation summary for completed rows, including normal train-1-to-25 logic/NL baselines. Formal joint is citation-free formal validity; translated joint is NL-to-FOL translated validity.}
\end{figure}
"""
    shortcut_kind_line_figure_block = ""
    if not shortcut_kind_rows.empty and (FIG_DIR / "ablation_shortcut_kind_rate_lines_vs_main.pdf").exists():
        shortcut_kind_line_figure_block = r"""
\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{figures/ablation_shortcut_kind_rate_lines_vs_main.pdf}
\caption{Shortcut-kind ablation as rate curves with the normal train-1-to-25 shortcut-neutral baselines at rate 0. Evaluation is shortcut-neutral for all rows.}
\end{figure}
"""
    shortcut_kind_eval_count = int(shortcut_kind_rows["n"].sum()) if not shortcut_kind_rows.empty else 0
    if shortcut_kind_eval_count >= 24:
        shortcut_kind_status_sentence = (
            "At this report generation time, SFT and shortcut-kind eval are complete; "
            "all eval artifacts are summarized below."
        )
        shortcut_kind_exec_sentence = "The shortcut-kind eval is complete at 24/24 JSONs."
    elif shortcut_kind_eval_count:
        shortcut_kind_status_sentence = (
            "At this report generation time, SFT is complete and shortcut-kind eval is running; "
            "the completed eval artifacts are summarized below."
        )
        shortcut_kind_exec_sentence = (
            f"The shortcut-kind eval has started and has {shortcut_kind_eval_count}/24 JSONs so far."
        )
    else:
        shortcut_kind_status_sentence = (
            "At this report generation time, SFT is complete and the eval has started, "
            "but no shortcut-kind eval JSON has been produced."
        )
        shortcut_kind_exec_sentence = "The shortcut-kind eval has started but has not written JSON yet."
    trace_control_eval_count = int(trace_control_rows["n"].sum()) if not trace_control_rows.empty else 0
    pseudocode_done = 0
    shuffled_nl_done = 0
    if not trace_control_rows.empty:
        by_template = trace_control_rows.set_index("template")
        if "pseudocode" in by_template.index:
            pseudocode_done = int(by_template.loc["pseudocode", "n"])
        if "shuffled_nl" in by_template.index:
            shuffled_nl_done = int(by_template.loc["shuffled_nl", "n"])
    if trace_control_eval_count >= 18:
        trace_control_exec_sentence = "Trace controls are complete at 18/18 rows."
        trace_control_status_sentence = "All trace-control eval rows are now complete."
    elif pseudocode_done < 3 and shuffled_nl_done >= 3:
        trace_control_exec_sentence = (
            f"Trace controls are partially evaluated at {trace_control_eval_count}/18 rows."
        )
        trace_control_status_sentence = (
            "The remaining trace-control row is \\texttt{pseudocode} seed 3409; "
            "\\texttt{shuffled\\_nl} is now three-seed complete."
        )
    else:
        trace_control_exec_sentence = (
            f"Trace controls are partially evaluated at {trace_control_eval_count}/18 rows."
        )
        trace_control_status_sentence = (
            "Current replacement rows are still finishing \\texttt{pseudocode} seed 3409 "
            "and \\texttt{shuffled\\_nl} seeds 3408/3409."
        )
    think_formal_train25_n = 0
    formal_think_complete_max = 0
    if not hybrid_order_rows.empty:
        match = hybrid_order_rows[
            (hybrid_order_rows["mode"] == "think_formal")
            & (hybrid_order_rows["train_max"] == 25)
        ]
        if not match.empty:
            think_formal_train25_n = int(match.iloc[0]["n"])
        formal_complete = hybrid_order_rows[
            (hybrid_order_rows["mode"] == "formal_think") & (hybrid_order_rows["n"] >= 3)
        ]
        if not formal_complete.empty:
            formal_think_complete_max = int(formal_complete["train_max"].max())
    formal_think_status = "\\texttt{formal\\_think} is still incomplete."
    if formal_think_complete_max:
        if formal_think_complete_max >= 25:
            formal_think_status = (
                "\\texttt{formal\\_think} is complete through train-1-to-25, "
                "so the full hybrid-order grid is complete."
            )
        else:
            formal_think_status = (
                f"\\texttt{{formal\\_think}} is complete through train-1-to-{formal_think_complete_max}, "
                "with deeper rows still incomplete."
            )
    if think_formal_train25_n >= 3:
        hybrid_complete = formal_think_complete_max >= 25
        hybrid_order_status_sentence = (
            "Current completed rows cover all \\texttt{think\\_formal} seeds through "
            f"train-1-to-25; {formal_think_status}"
        )
        hybrid_order_table_caption = (
            ("Hybrid-order full-grid summary. " if hybrid_complete else "Hybrid-order partial summary. ")
            + "\\texttt{think\\_formal} train-1-to-25 is three-seed complete; "
            + formal_think_status
        )
        hybrid_order_figure_caption = (
            ("Hybrid order full eval with normal logic/NL baseline curves. " if hybrid_complete else "Hybrid order partial eval with normal logic/NL baseline curves. ")
            + "\\texttt{think\\_formal} is NL then formal and is complete through train-1-to-25; "
            + formal_think_status
        )
    else:
        hybrid_order_status_sentence = (
            "Current completed rows cover all \\texttt{think\\_formal} seeds through "
            "train-1-to-20; train-1-to-25 is partial, and \\texttt{formal\\_think} "
            "is still incomplete."
        )
        hybrid_order_table_caption = (
            "Hybrid-order partial summary. \\texttt{think\\_formal} train-1-to-25 "
            "is still partial; \\texttt{formal\\_think} remains pending."
        )
        hybrid_order_figure_caption = (
            "Hybrid order partial eval with normal logic/NL baseline curves. \\texttt{think\\_formal} is NL then formal "
            "and is complete through train-1-to-20, with train-1-to-25 partial; "
            "\\texttt{formal\\_think} is still pending."
        )
    hybrid_order_figure_block = ""
    if not hybrid_order_rows.empty and (FIG_DIR / "ablation_hybrid_order_partial.pdf").exists():
        hybrid_order_figure_block = r"""
\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{figures/ablation_hybrid_order_partial.pdf}
\caption{""" + hybrid_order_figure_caption + r"""}
\end{figure}
"""
    conditioned_50k_curve_block = ""
    if (FIG_DIR / "ablation_conditioned_dual_50k_convergence_train1to25.pdf").exists():
        conditioned_50k_curve_block = r"""
\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{figures/ablation_conditioned_dual_50k_convergence_train1to25.pdf}
\caption{Conditioned dual-modality 50k convergence curves for train-1-to-25. Points are three-seed means for completed checkpoint evals; both conditioned-logic and conditioned-NL checkpoint curves now reach 50k.}
\end{figure}
"""
    paired_full_figure_block = ""
    if not paired_full_rows.empty and (FIG_DIR / "paired_full_suite_official_igsm_partial.pdf").exists():
        paired_full_figure_block = r"""
\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{figures/paired_full_suite_official_igsm_partial.pdf}
\caption{Partial official-iGSM full-suite readout from currently completed paired eval rows. Inline n annotations mark any train-depth slices that are not yet three-seed complete.}
\end{figure}
"""
    paired_full_family_figure_block = ""
    if not paired_full_rows.empty and (FIG_DIR / "paired_full_suite_family_partial.pdf").exists():
        paired_full_family_figure_block = r"""
\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{figures/paired_full_suite_family_partial.pdf}
\caption{Partial paired-family readout for completed eval rows. Empty panels mark family/train-depth slices with no completed JSON yet; inline n annotations mark slices that are not yet three-seed complete.}
\end{figure}
"""
    paired_igsm_semantic_figure_block = ""
    if not paired_igsm_semantic_rows.empty and (FIG_DIR / "paired_igsm_semantic_summary.pdf").exists():
        paired_igsm_semantic_figure_block = r"""
\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{figures/paired_igsm_semantic_summary.pdf}
\caption{Corrected semantic iGSM curves over train-depth range. For logic, valid/parse is internal citation-free proof validity; for NL, valid/parse is NL-to-logic parse coverage after semantic alias canonicalization.}
\end{figure}
"""
    olmo_ckpt_figures = "\n".join(
        f"""\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/olmo7b_checkpoint_train1to{train_max}_correct16.pdf}}
\\caption{{Seed-3407 train-1-to-{train_max} checkpoint curves for correct@16.}}
\\end{{figure}}
\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/olmo7b_checkpoint_train1to{train_max}_joint16.pdf}}
\\caption{{Seed-3407 train-1-to-{train_max} checkpoint curves for joint correct+valid@16.}}
\\end{{figure}}"""
        for train_max in train_maxes
    )
    tiny_band_figures = "\n".join(
        f"""\\begin{{figure}}[H]\\centering
\\includegraphics[width=0.72\\linewidth]{{figures/tiny_llama_{size}_bands_correct_k8.pdf}}
\\caption{{Tiny Llama {size} train/OOD/hard-tail/depth-50 correct@8 by template.}}
\\end{{figure}}
\\begin{{figure}}[H]\\centering
\\includegraphics[width=0.72\\linewidth]{{figures/tiny_llama_{size}_bands_joint_k8.pdf}}
\\caption{{Tiny Llama {size} train/OOD/hard-tail/depth-50 joint correct+valid@8 by template.}}
\\end{{figure}}"""
        for size in tiny_sizes
    )
    tiny_depth_figures = "\n".join(
        f"""\\begin{{figure}}[H]\\centering
\\includegraphics[width=0.72\\linewidth]{{figures/tiny_llama_{size}_depth_correct_k8.pdf}}
\\caption{{Tiny Llama {size} seed-mean depth curve for correct@8. The dotted line marks train max depth 10.}}
\\end{{figure}}
\\begin{{figure}}[H]\\centering
\\includegraphics[width=0.72\\linewidth]{{figures/tiny_llama_{size}_depth_joint_k8.pdf}}
\\caption{{Tiny Llama {size} seed-mean depth curve for joint correct+valid@8. The dotted line marks train max depth 10.}}
\\end{{figure}}"""
        for size in tiny_sizes
    )
    tiny_ckpt_figures = "\n".join(
        f"""\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/tiny_llama_{size}_checkpoint_correct_k8.pdf}}
\\caption{{Tiny Llama {size} seed-mean checkpoint curves for correct@8.}}
\\end{{figure}}
\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/tiny_llama_{size}_checkpoint_joint_k8.pdf}}
\\caption{{Tiny Llama {size} seed-mean checkpoint curves for joint correct+valid@8.}}
\\end{{figure}}"""
        for size in tiny_sizes
    )
    tiny_ckpt_depthband_figures = "\n".join(
        f"""\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/tiny_llama_{size}_checkpoint_depthbands_correct_k8.pdf}}
\\caption{{Tiny Llama {size} optimizer-step curves for correct@8 over eval-depth bands.}}
\\end{{figure}}
\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/tiny_llama_{size}_checkpoint_depthbands_joint_k8.pdf}}
\\caption{{Tiny Llama {size} optimizer-step curves for joint correct+valid@8 over eval-depth bands.}}
\\end{{figure}}"""
        for size in tiny_sizes
    )
    tiny_100k_ckpt_depthband_figures = "\n".join(
        f"""\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/tiny_llama_100k_{size}_checkpoint_depthbands_correct_k8.pdf}}
\\caption{{Tiny Llama 100k {size} optimizer-step curves for correct@8 over eval-depth bands.}}
\\end{{figure}}
\\begin{{figure}}[H]\\centering
\\includegraphics[width=\\linewidth]{{figures/tiny_llama_100k_{size}_checkpoint_depthbands_joint_k8.pdf}}
\\caption{{Tiny Llama 100k {size} optimizer-step curves for joint correct+valid@8 over eval-depth bands.}}
\\end{{figure}}"""
        for size in tiny_sizes
    )

    delta_label = "$\\Delta$ vs logic"
    corrected_status_sentence = (
        "The corrected 30-run grid has passed its artifact and qualitative gates; "
        "replacement evidence is reported in the next section."
        if corrected_branchproof_ready
        else "The corrected grid remains gated and is not yet used as evidence."
    )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\title{{Formal Logic CoT Synthetic Results Update}}
\date{{2026-07-23}}
\begin{{document}}
\maketitle

\section{{Critical BranchProof audit status}}
All pre-2026-07-10 HFSA/BranchProof results above depth 17 in this report are
quarantined historical artifacts. A forward-closure audit found that the old
generator wrapped constants after 18 layers, creating multiple derivable
candidate answers in 73/96 audited depth-20 examples and 92/96 audited
depth-40/45/50 examples. The defect affects the main depth-scaling result and
every architecture, syntax, shortcut, hybrid, conditioned-dual, batch-size,
tiny-pretraining, downstream-transfer, and midtraining artifact trained from
that construction. Do not cite these numbers. The corrected generator uses
fresh constants at every layer, passes a unique-answer closure gate, and uses
equal logic/NL generation caps plus greedy and pass@1 metrics.
{corrected_status_sentence} The independent AttrCon results are not affected.
Full evidence and recovery jobs are recorded in
\texttt{{docs/branchproof\_uniqueness\_audit\_2026-07-10.md}}.

{corrected_branchproof_block}
{selected_branchproof_block}

\section{{Executive insights}}
\begin{{itemize}}
\item {corrected_branchproof_executive}
\item {selected_branchproof_executive if selected_branchproof_ready else "Selected corrected BranchProof controls remain gated."}
\item AttrCon remains independent positive-but-mixed evidence: formal traces improve mean answer correctness, while natural traces have the higher joint-validity mean.
\item The corrected Qwen2.5 continual-pretraining pilot remains null or mixed after raw-generation and truncation audits, so the broader mixture grid is rejected.
\item All later sections based on the wrapped-constant BranchProof construction are retained only as quarantined provenance. Only the audited corrected controls summarized above are evidence.
\end{{itemize}}

\section{{Metric note for OOD benchmarks}}
EM means exact match: after extracting explicit answer content from an \texttt{{<answer>}} tag or answer marker, the evaluator normalizes the prediction and gold answer and assigns 1 only when they exactly match, then averages over examples. GSM8K is numeric EM and is effectively an accuracy over single numeric answers, but the report does not fall back to arbitrary numbers elsewhere in a generated trace. For HotpotQA, 2WikiMultiHopQA, and MuSiQue the answers are free-form strings with aliases and partial-overlap possibilities, so the standard report is EM plus token-level F1 rather than calling the result plain accuracy. F1 gives partial credit for overlapping answer tokens; EM is the stricter all-or-nothing string match.

\section{{Training sequence token lengths}}
This audit uses the OLMo tokenizer over the actual SFT training mixtures. NL targets are longer than logic targets at every train range; total sequence lengths include the prompt plus target.

\begin{{table}}[H]
\centering
\small
{latex_table(token_length_rows, [
    ("train_range", "train range"),
    ("logic_target", "logic target"),
    ("nl_target", "NL target"),
    ("logic_total", "logic total"),
    ("nl_total", "NL total"),
])}
\caption{{Mean token lengths for main OLMo-7B SFT sequences by train-depth range.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{latex_table(symbol_padded_token_rows, [
    ("condition", "condition"),
    ("n", "n"),
    ("target_mean", "target mean"),
    ("target_p95", "target p95"),
    ("total_mean", "total mean"),
    ("total_p95", "total p95"),
    ("target_vs_nl", "target/NL"),
    ("total_vs_nl", "total/NL"),
    ("truncation_rate", "trunc."),
])}
\caption{{Token-length match for logic length-control ablations on the same 512 train-1-to-25 examples. Symbol-padded lengthens formal atoms mechanically; wordified logic uses natural attribute predicate names while preserving formal proof rules.}}
\end{{table}}

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(ablation_token_rows, [
    ("condition", "condition"),
    ("n", "n"),
    ("prompt_mean", "prompt"),
    ("target_mean", "target"),
    ("target_p95", "target p95"),
    ("total_mean", "total"),
    ("total_p95", "total p95"),
    ("proof_mean", "proof/body"),
    ("syntax_occ_mean", "syntax occ."),
    ("syntax_tok_mean", "syntax tok."),
    ("tok_per_syntax_occ", "tok/syntax"),
    ("syntax_tok_share", "syntax share"),
])}
\caption{{Ablation token audit on 512 deterministic train-1-to-25 examples with the OLMo tokenizer. The syntax/operator columns count occurrences and tokenized length of a fixed audit lexeme set: proof tags, formal operators, rule labels, pseudocode markers, and common derivation words. They are diagnostics, not a model-internal category. In this generator, \texttt{{terse\_nl}} currently matches \texttt{{nl\_exact}} token-for-token because default HFSA NL proof lines are already short.}}
\end{{table}}

\section{{Main OLMo-7B downstream OOD}}
The broad OOD run has completed the 30-row main OLMo grid. This is downstream transfer under the strict answer extractor, not a synthetic validity score. The headline pattern is split: NL is much better on GSM8K numeric EM, while logic is much better on the context-provided multi-hop QA F1/EM tasks. Bold values mark the best value in each metric column.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(main_ood_rows, [
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("gsm8k_em", "GSM8K EM"),
    ("hotpot_em", "Hotpot EM"),
    ("twowiki_em", "2Wiki EM"),
    ("musique_em", "MuSiQue EM"),
], bold_columns={"gsm8k_em", "hotpot_em", "twowiki_em", "musique_em"})}
\caption{{OOD exact-match means over seeds for the main OLMo-7B checkpoints.}}
\end{{table}}

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(main_ood_rows, [
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("hotpot_f1", "Hotpot F1"),
    ("twowiki_f1", "2Wiki F1"),
    ("musique_f1", "MuSiQue F1"),
], bold_columns={"hotpot_f1", "twowiki_f1", "musique_f1"})}
\caption{{OOD F1 means over seeds for the main OLMo-7B checkpoints. GSM8K is omitted because the task is scored by numeric exact match rather than answer-token F1.}}
\end{{table}}

\subsection{{Bare-format OOD rerun}}
The format-matched bare rerun wraps the task content in \texttt{{<question>...</question>}} and lets the checkpoint emit its learned answer format. The 30-row main OLMo-7B slice is complete. Compared with the answer-only OOD run, NL remains much stronger on GSM8K, while logic remains stronger on context-provided QA.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(main_cot_bare_rows, [
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("gsm8k_em", "GSM8K EM"),
    ("hotpot_em", "Hotpot EM"),
    ("twowiki_em", "2Wiki EM"),
    ("musique_em", "MuSiQue EM"),
], bold_columns={"gsm8k_em", "hotpot_em", "twowiki_em", "musique_em"})}
\caption{{Bare-format OOD exact-match means for the completed main OLMo-7B slice.}}
\end{{table}}

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(main_cot_bare_rows, [
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("hotpot_f1", "Hotpot F1"),
    ("twowiki_f1", "2Wiki F1"),
    ("musique_f1", "MuSiQue F1"),
], bold_columns={"hotpot_f1", "twowiki_f1", "musique_f1"})}
\caption{{Bare-format OOD F1 means for the completed main OLMo-7B slice.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{latex_table(olmo32_cot_bare_gsm8k, [
    ("template", "template"),
    ("train_max", "train max"),
    ("seed", "seed"),
    ("gsm8k_em", "GSM8K EM"),
    ("gsm8k_tag", "GSM8K tag"),
], bold_columns={"gsm8k_em", "gsm8k_tag"})}
\caption{{OLMo-2-32B bare-format GSM8K rerun. Full LongBench is intentionally skipped for this model because its configured context limit is 4096.}}
\end{{table}}

\section{{Main OLMo-7B HFSA result}}
\begin{{table}}[H]
\centering
\small
{latex_table(main_rows, [
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("hard_tail_correct@16", "hard c@16"),
    ("hard_tail_joint@16", "hard joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Final sparse-protocol means over seeds. Logic joint uses citation-free formal validity; NL joint uses translated NL-to-FOL validity.}}
\end{{table}}

\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{figures/olmo7b_final_by_train_depth.pdf}}
\caption{{Final OOD and hard-tail correct/joint performance as train depth increases.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{figures/olmo7b_depth_correct16.pdf}}
\caption{{Depth curves for final correct@16, averaged over seeds. Vertical dotted line marks the max train depth.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{figures/olmo7b_depth_joint16.pdf}}
\caption{{Depth curves for final joint correct+valid@16, averaged over seeds.}}
\end{{figure}}
The checkpoint figures below compare each matched logic/NL train-depth pair separately and report only @16.
{main_ckpt_note}
{olmo_ckpt_figures}
The next two figures slice the train-1-to-25 checkpoint curve by eval-depth bands. The intermediate protocol contains depths 30, 40, and 50, but not depths 35 or 45, so exact 30--35 or 36--40 curves would require an additional focused checkpoint eval.
\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{figures/olmo7b_checkpoint_train1to25_depthbands_correct16.pdf}}
\caption{{Seed-3407 train-1-to-25 optimizer-step curves for correct@16 over available eval-depth bands.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=\linewidth]{{figures/olmo7b_checkpoint_train1to25_depthbands_joint16.pdf}}
\caption{{Seed-3407 train-1-to-25 optimizer-step curves for joint correct+valid@16 over available eval-depth bands.}}
\end{{figure}}

\section{{Tiny Llama scratch-pretraining result}}
These small random-init models were not expected to solve depth-50 extrapolation. The useful signal is that logic is consistently stronger than matched NL on answer-only OOD pass@8, especially at 200M, while strict joint validity remains zero.

\begin{{table}}[H]
\centering
\small
{latex_table(tiny_rows, [
    ("size", "size"),
    ("template", "template"),
    ("train_correct@8", "train c@8"),
    ("ood_correct@8", "OOD c@8"),
    ("hard_tail_correct@8", "hard c@8"),
    ("depth50_correct@8", "d50 c@8"),
    ("ood_joint@8", "OOD joint@8"),
])}
\caption{{Tiny Llama final sparse pass@8 metrics.}}
\end{{table}}

\subsection{{Tiny Llama 100k-step rerun}}
The 100k-step rerun is complete for all three seeds, sizes, and templates. It does not solve strict depth-50 extrapolation; the notable change is that the 100M NL model improves answer-only OOD correct@8 relative to its 20k run, while joint validity remains zero.

\begin{{table}}[H]
\centering
\small
{latex_table(tiny_100k_rows, [
    ("size", "size"),
    ("template", "template"),
    ("train_correct@8", "train c@8"),
    ("ood_correct@8", "OOD c@8"),
    ("hard_tail_correct@8", "hard c@8"),
    ("depth50_correct@8", "d50 c@8"),
    ("ood_joint@8", "OOD joint@8"),
    ("depth50_joint@8", "d50 joint@8"),
])}
\caption{{Tiny Llama 100k-step final sparse pass@8 metrics.}}
\end{{table}}

Tiny 100k checkpoint pass@k rows available at report generation time: {tiny_100k_ckpt_count}; the completed grid contains checkpoints 20k, 40k, 60k, 80k, and 100k for every size/template/seed.
{tiny_100k_ckpt_depthband_figures}

The tiny plots separate model sizes and separate answer correctness from joint correct+valid.
{tiny_band_figures}
{tiny_depth_figures}
"""
    if tiny_ckpt_count:
        tex += (
            f"\nTiny checkpoint pass@k rows available at report generation time: {tiny_ckpt_count}. "
            "The curves below include all available intermediate checkpoint rows plus the final 20k checkpoint.\n"
        )
        tex += "\n" + tiny_ckpt_figures + "\n"
        tex += "\n" + tiny_ckpt_depthband_figures + "\n"
    else:
        tex += (
            "\nCheckpoint pass@k files for tiny Llama were not present when this report was generated; "
            "the report builder will include checkpoint curves automatically once the submitted intermediate eval finishes.\n"
        )

    tex += rf"""
\subsection{{Tiny Llama downstream OOD eval}}
The tiny downstream OOD evals completed with the 8192-context fallback. LongBench contexts are truncated for these tiny models, so these tables are smoke/result-sanity readouts rather than fair long-context QA claims.

\begin{{table}}[H]
\centering
\small
{latex_table(tiny_ood_summary, [
    ("size", "size"),
    ("template", "template"),
    ("n", "n"),
    ("gsm8k_em", "GSM8K EM"),
    ("gsm8k_tag", "GSM8K tag"),
    ("hotpot_em", "Hotpot EM"),
    ("twowiki_em", "2Wiki EM"),
    ("musique_em", "MuSiQue EM"),
], bold_columns={"gsm8k_em", "gsm8k_tag", "hotpot_em", "twowiki_em", "musique_em"})}
\caption{{Tiny Llama OOD exact-match metrics after strict answer extraction.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{latex_table(tiny_ood_summary, [
    ("size", "size"),
    ("template", "template"),
    ("n", "n"),
    ("hotpot_f1", "Hotpot F1"),
    ("twowiki_f1", "2Wiki F1"),
    ("musique_f1", "MuSiQue F1"),
], bold_columns={"hotpot_f1", "twowiki_f1", "musique_f1"})}
\caption{{Tiny Llama OOD F1 metrics after strict answer extraction. GSM8K is omitted because it is scored by numeric exact match.}}
\end{{table}}

\subsection{{Tiny Llama bare-format OOD reruns}}
Both the 20k and 100k tiny bare-format OOD arrays completed. All strict EM/F1 values remain zero on GSM8K and LongBench; the only movement is answer-tag adherence, mostly in the 200M checkpoints.

\begin{{table}}[H]
\centering
\small
{latex_table(tiny_cot_bare_ood_summary, [
    ("size", "size"),
    ("template", "template"),
    ("n", "n"),
    ("gsm8k_em", "GSM8K EM"),
    ("gsm8k_tag", "GSM8K tag"),
    ("hotpot_em", "Hotpot EM"),
    ("twowiki_em", "2Wiki EM"),
    ("musique_em", "MuSiQue EM"),
], bold_columns={"gsm8k_em", "gsm8k_tag", "hotpot_em", "twowiki_em", "musique_em"})}
\caption{{Tiny Llama 20k bare-format OOD exact-match metrics.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{latex_table(tiny_100k_cot_bare_ood_summary, [
    ("size", "size"),
    ("template", "template"),
    ("n", "n"),
    ("gsm8k_em", "GSM8K EM"),
    ("gsm8k_tag", "GSM8K tag"),
    ("hotpot_em", "Hotpot EM"),
    ("twowiki_em", "2Wiki EM"),
    ("musique_em", "MuSiQue EM"),
], bold_columns={"gsm8k_em", "gsm8k_tag", "hotpot_em", "twowiki_em", "musique_em"})}
\caption{{Tiny Llama 100k bare-format OOD exact-match metrics.}}
\end{{table}}

\section{{Architecture ablations}}
The architecture comparison now includes completed Qwen-2.5-1.5B, Qwen-2.5-7B, and Gemma-3-4B sparse evals, plus the main OLMo-7B baseline for the overlapping train ranges. OLMo-2-32B is shown only as a short-context slice because its real context limit prevents the depth-50 protocol.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(architecture_rows, [
    ("model", "model"),
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth30_50_correct@16", "d30--50 c@16"),
    ("depth30_50_joint@16", "d30--50 joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
    ("shortctx_correct@16", "short c@16"),
    ("shortctx_joint@16", "short joint@16"),
])}
\caption{{Architecture sparse eval comparison. Depth 30--50 averages depths 30, 35, 40, 45, and 50 when available; short-context averages depths 1, 2, 5, 10, 12, and 15 and is the only valid OLMo-2-32B synthetic slice.}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/architecture_ood_correct16_by_train_depth.pdf}}
\caption{{Architecture comparison for OOD correct@16.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/architecture_depth30_50_correct16_by_train_depth.pdf}}
\caption{{Architecture comparison for eval-depth 30--50 correct@16.}}
\end{{figure}}

\section{{Active experiment status}}
This table is artifact-based at report-generation time: it counts completed SFT final adapters where applicable and completed sparse pass@k JSONs. Live Slurm state is maintained in \texttt{{docs/running\_experiments.md}}.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(experiment_status_rows, [
    ("experiment", "experiment"),
    ("scope", "scope"),
    ("sft_done", "SFT done"),
    ("sft_expected", "SFT expected"),
    ("eval_done", "eval done"),
    ("eval_expected", "eval expected"),
    ("status", "status"),
])}
\caption{{Artifact status for paired-family repeats and active ablation families.}}
\end{{table}}

\subsection{{Fresh paired rerun readouts}}
The fresh paired reruns are now complete for semantic iGSM, typed maze, and hard attribute. The typed-maze table is from the patched typed-symbol rerun, not the stale old maze root. It uses a formal max-new-token cap of 4096 because the earlier 8192-token formal eval could not finish within the A100 walltime; NL remains capped at 6144. The hard-attribute rows are from the fresh hard-attribute-only eval under \texttt{{\$HPCVAULT}}. Hard-attribute logic validity is meaningful under citation-free formal checking; hard-attribute NL validity remains unsupported because the generic NL-to-FOL translator does not yet parse this family, so the NL joint and parse columns should not be used as negative evidence about the model. Typed maze is a clear negative result after the typed-symbol fix: shallow train-band traces are valid, but OOD and depth-50 remain essentially unsolved. Representative train-1-to-25 samples show the main failure mode is not answer extraction on otherwise complete traces: formal depth-25/50 generations spend the budget on constants, premises, and partial derivations and often never emit \texttt{{<answer>}}, while NL depth-25/50 generations copy natural-language premise chains through roughly move 18--20 and also omit the answer tag.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(active_paired_partial_rows, [
    ("display_family", "family"),
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("train_correct@16", "train c@16"),
    ("train_joint@16", "train joint@16"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_valid_or_parse@16", "OOD valid/parse@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Fresh typed-maze and hard-attribute results. For logic, valid/joint means citation-free formal validity; for NL, valid/parse is NL-to-FOL parse coverage and joint is translated-valid joint.}}
\end{{table}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/active_paired_partial_summary.pdf}}
\caption{{Fresh typed-maze and hard-attribute curves. Inline n annotations show seed coverage.}}
\end{{figure}}

The batch-size ablation is complete for the planned seed-3407 diagnostic grid: single-modality logic, single-modality NL, and 50--50 conditioned-dual training at effective batch sizes 2, 4, 8, and 16. Batch size 16 should be read as effective batch 16, implemented by microbatch 8 and gradient accumulation 2 after true physical bsz16 OOMed on A100-80GB. The result is non-monotonic and does not support the simple hypothesis that larger stratified batches rescue conditioned dual. Conditioned-NL is strongest at bsz2 and worsens at larger batches on OOD joint; conditioned-logic recovers at bsz16 but is still not a clean monotone batch-size effect. Because this is one seed, it is a diagnostic ablation rather than final causal evidence.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(batch_size_partial_rows, [
    ("train_condition", "train"),
    ("eval_condition", "eval"),
    ("batch_size_label", "bsz"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_valid_or_parse@16", "OOD valid/parse@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{HFSA batch-size ablation results. Batch size 16 is effective batch 16 implemented as microbatch 8 with gradient accumulation 2.}}
\end{{table}}

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(batch_size_diagnostic_rows, [
    ("train_condition", "train"),
    ("eval_condition", "eval"),
    ("best_ood_joint_bsz", "best OOD bsz"),
    ("best_ood_joint@16", "best OOD joint@16"),
    ("best_depth50_joint_bsz", "best d50 bsz"),
    ("best_depth50_joint@16", "best d50 joint@16"),
    ("ood_joint_range", "OOD joint range"),
    ("depth50_joint_range", "d50 joint range"),
    ("mean_ood_joint_delta_vs_single", "mean OOD delta vs single"),
    ("best_ood_joint_delta_vs_single", "best OOD delta vs single"),
])}
\caption{{Distilled batch-size diagnostics. Delta columns compare conditioned-dual rows to the matched single-modality row at the same effective batch size; blank deltas denote the single-modality baselines themselves.}}
\end{{table}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/hfsa_batch_size_ablation_partial.pdf}}
\caption{{HFSA batch-size ablation curves over effective batch size. Solid lines are single-modality runs; dashed lines are conditioned-dual eval modes.}}
\end{{figure}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.85\linewidth]{{figures/hfsa_batch_size_conditioned_delta.pdf}}
\caption{{Conditioned-dual joint-validity deltas relative to matched single-modality runs at the same effective batch size. Positive means conditioned dual is better; negative means it is worse.}}
\end{{figure}}

Representative sample inspection is consistent with the aggregate story. The conditioned prompts are correct: formal-mode rows use \texttt{{<reasoning\_mode>formal\_logic</reasoning\_mode>}} and \texttt{{<formal>}}, while NL-mode rows use natural-language traces. The failures are not a train--test prompt mismatch. Instead, high-depth formal samples often emit long constant/predicate/premise blocks and then lose strict cited/grounded validity; citation-free validity can still be high when the formulas are recoverable without citations. High-depth NL samples preserve the \texttt{{<think>}} surface but copy long premise lists, drift, or truncate before a translated proof/answer is complete. The bsz2 conditioned rows sometimes beat the matched single-modality row, but the effect is not stable across batch sizes or depth-50, so the safest conclusion is that batch composition alone does not explain the conditioned-dual gap.

\section{{Old paired-family full-suite diagnostics}}
The old combined paired-family eval remains partial and scientifically stale for iGSM semantic grounding and maze typed-symbol claims. Use the fresh semantic iGSM, typed-maze, and hard-attribute rows above for current conclusions. The old root is still useful as a historical diagnostic: \texttt{{official\_igsm}} completed, \texttt{{maze\_navigation}} has only early rows, and the old combined hard attribute-constraint rows have no completed eval JSONs. The \texttt{{joint}} columns are template-specific: citation-free internal formal validity for logic and translated NL-to-FOL validity for \texttt{{nl\_exact}}. Old maze train-1-to-5 already showed a sharp train/OOD split rather than a bad gold generator; the fresh typed-maze completion confirms the same qualitative failure after fixing the room/key namespace. For iGSM, grounded joint validity is currently zero away from trivial retrieval cases because generated variable names and arithmetic-substitution citations do not reliably align with the canonical grounded checker. The targeted iGSM NL rerun recovered near-complete parser coverage, but generated NL translated-validity is still zero on OOD/depth-50 slices; use correctness and parser coverage as diagnostics unless grounded/canonical checks are improved.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(paired_full_rows, [
    ("family", "family"),
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("train_correct@16", "train c@16"),
    ("train_joint@16", "train joint@16"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("ood_grounded_joint@16", "OOD grounded joint"),
    ("ood_nl_parse@16", "OOD NL parse"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Partial paired full-suite summary for completed eval JSONs.}}
\end{{table}}
{paired_full_figure_block}
{paired_full_family_figure_block}

\subsection{{Semantic iGSM rerun}}
The semantic iGSM rerun is the current iGSM result: it uses bare semantic variables and definition-style NL proof prose rather than the old hidden \texttt{{v\_}} handles and generic ``official iGSM relation'' wording. The table below uses the corrected NL alias canonicalizer and clean forced NL re-eval. For logic, ``valid/parse'' is internal citation-free proof validity; for NL it is NL-to-logic parse coverage. The corrected result separates answer accuracy from validated reasoning: NL answer accuracy rises strongly with train range, but OOD/depth-50 translated joint validity remains zero because long generated traces drift/truncate and only small prefixes validate. Logic has nonzero internal validity but lower answer accuracy and zero strict grounded joint.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(paired_igsm_semantic_rows, [
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_valid_or_parse@16", "OOD valid/parse@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("ood_grounded_joint@16", "OOD grounded joint"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_valid_or_parse@16", "d50 valid/parse@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Corrected semantic iGSM pass@16 summary over three seeds. OOD is the hard-tail band.}}
\end{{table}}
{paired_igsm_semantic_figure_block}

\section{{Targeted ablations}}
Completed ablation results are shown here; active arrays are tracked in the handoff documents. The current same-token-budget experiment is more precisely a same target-token budget experiment. The logic row in that experiment is a control rerun at 10k steps; the NL row is shortened to 7140 steps so target-token exposure approximately matches 10k logic steps. It does not match total prompt-plus-target tokens; that would require roughly 8600 NL steps and has not been run yet. The symbol-padded logic control is the completed total-sequence-length match to NL at the same optimizer-step budget; the wordified logic length-control is the cleaner equal-length formal follow-up and is now complete. Shortcut rates 0.3, 0.5, and 0.8 are complete. Conditioned dual-modality 10k eval is complete, and the 50k continuation is running.

The shortcut-rate ablation changes only the training distribution. With probability equal to the shortcut rate, a training example uses the \texttt{{hard\_fsa\_schema}} shortcut generator: the gold path obeys a shared marker-conditioned transition schema and carries redundant marker facts that make a family-level transition heuristic predictive. Non-gold branches remain coherent but do not follow that schema. Evaluation always resets \texttt{{shortcut\_rate=0.0}}, so these rows test whether training on shortcut-rich traces hurts or helps transfer back to shortcut-neutral depth extrapolation.

\begin{{table}}[H]
\centering
\small
{latex_table(token_budget_exposure_rows, [
    ("condition", "condition"),
    ("status", "status"),
    ("steps", "steps"),
    ("target_tok_per_ex", "target tok/ex"),
    ("target_exposure_vs_logic", "target exposure"),
    ("total_tok_per_ex", "total tok/ex"),
    ("total_exposure_vs_logic", "total exposure"),
])}
\caption{{Token-budget accounting for train-1-to-25. Target-token ratios use the 512-example audit that set the submitted step count; total-token ratios use the report token-length audit. Exposure columns are relative to 10k logic steps.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{latex_table(symbol_padded_eval_rows, [
    ("condition", "condition"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Logic length-control ablations compared with the main train-1-to-25 runs.}}
\end{{table}}

The length-control formal variants are not broken gold targets: focused tests and generated examples validate the symbol-padded and wordified training traces. They are worse because the representation changes the model-facing problem. Symbol padding turns compact atoms such as \texttt{{Fa}} into predicate-call forms such as \texttt{{PF(ca)}}, adding punctuation and multi-token syntax pieces. Wordified logic keeps formal rules but expands predicate symbols into natural attribute names such as \texttt{{North(a)}} and \texttt{{Teal(a)}}, so the model loses the compact symbolic alphabet while still needing formal proof discipline.

\subsection{{Length-control surface examples}}
{latex_length_control_surface_examples(length_control_surface_rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/ablation_logic_length_control_depth_curve_train1to25.pdf}}
\caption{{Train-1-to-25 depth curve comparing compact logic, length-control logic variants, and main NL exact. Curves are seed means over the sparse final eval depths; the dotted line marks the maximum training depth.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\small
{latex_table(token_budget_comparison_rows, [
    ("condition", "condition"),
    ("steps", "steps"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("delta_ood_correct@16", delta_label),
    ("ood_joint@16", "OOD joint@16"),
    ("delta_ood_joint@16", delta_label),
    ("depth50_correct@16", "d50 c@16"),
    ("delta_depth50_correct@16", delta_label),
    ("depth50_joint@16", "d50 joint@16"),
    ("delta_depth50_joint@16", delta_label),
])}
\caption{{Same target-token budget control compared directly to the main train-1-to-25 logic baseline. Delta columns subtract the main logic row.}}
\end{{table}}

\begin{{table}}[H]
\centering
\small
{latex_table(shortcut_comparison_rows, [
    ("template", "template"),
    ("shortcut_rate", "shortcut rate"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Shortcut-rate ablation, including the shortcut-neutral main train-1-to-25 baseline as rate 0.0. Eval remains shortcut-neutral for all rows.}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.9\linewidth]{{figures/ablation_shortcut_rate_vs_main.pdf}}
\caption{{Shortcut-rate ablation compared with shortcut-neutral main train-1-to-25 baselines.}}
\end{{figure}}

\subsection{{Other shortcut types}}
The shortcut-kind controls test two concrete shortcut mechanisms: a \texttt{{position}} shortcut, where the target path is statistically associated with position-like structure, and an \texttt{{initial\_marker}} shortcut, where an early marker fact is predictive of the target transition family. Both are trained at rates 0.5 and 0.8 for logic and NL over three seeds, with shortcut-neutral eval. {shortcut_kind_status_sentence}

\begin{{table}}[H]
\centering
\small
{latex_table(shortcut_kind_rows, [
    ("shortcut_kind", "shortcut kind"),
    ("template", "template"),
    ("shortcut_rate", "rate"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Shortcut-kind eval summary. No rows means the eval dependency has not started or has not written JSON yet.}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/ablation_shortcut_kind_summary.pdf}}
\caption{{Shortcut-kind ablation for position and initial-marker shortcuts. Eval prompts are shortcut-neutral; bars show three-seed means for completed rows.}}
\end{{figure}}
{shortcut_kind_line_figure_block}
Sample metadata checks over the shortcut-kind sample JSONLs found \texttt{{active\_branch\_first=None}} for completed eval rows, confirming that these evals are shortcut-neutral. The \texttt{{initial\_marker}} NL improvement at rate 0.8 is therefore not explained by leaving the shortcut active at eval time. It remains a surprising regularization/distribution result rather than clean evidence that increasing shortcut rate generally helps NL; the original schema shortcut-rate ablation still shows NL degradation as shortcut rate increases.

\subsection{{Trace controls}}
The trace-control ablation trains train-1-to-25 models with six altered trace styles, always over three seeds and evaluated on the same 1-to-50 sparse protocol. The six controls are: \texttt{{terse\_nl}} (intended shorter natural-language reasoning; currently token-identical to \texttt{{nl\_exact}} on HFSA), \texttt{{rule\_annotated\_nl}} (NL with explicit rule names), \texttt{{pseudocode}} (algorithm-like trace), \texttt{{shuffled\_logic}} (formal lines shuffled as a negative control), \texttt{{invalid\_logic}} (formally invalid proof trace negative control), and \texttt{{shuffled\_nl}} (NL trace order shuffled). The table includes the normal train-1-to-25 compact-logic and exact-NL baselines for direct comparison. Manual inspection found that the early \texttt{{rule\_annotated\_nl}} translated-validity metrics were an evaluator artifact: lines such as \texttt{{a is teal. [rule: R]}} were not stripped before controlled NL-to-FOL parsing. The translator now unwraps rule annotations and pseudocode \texttt{{derive "..."}}, and the completed repair rows are included in the table. {trace_control_status_sentence}

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(trace_control_comparison_rows, [
    ("template", "trace control"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_formal_joint@16", "OOD formal joint"),
    ("ood_translated_joint@16", "OOD translated joint"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_formal_joint@16", "d50 formal joint"),
    ("depth50_translated_joint@16", "d50 translated joint"),
])}
\caption{{Trace-control summary. Both formal and translated joint are shown because the controls intentionally move between formal, NL, and hybrid-like trace surfaces.}}
\end{{table}}
{trace_control_figure_block}

\subsection{{Ablation training-sequence examples}}
{latex_ablation_examples(ablation_example_rows)}

\subsection{{Hybrid order}}
The hybrid-order ablation trains a single prompt containing both trace substrates and one answer at the end. \texttt{{think\_formal}} means NL first, then formal logic; \texttt{{formal\_think}} means formal logic first, then NL. The full suite is train-1-to-5/10/15/20/25 over three seeds and eval 1-to-50. The table includes the normal compact-logic and exact-NL baselines at each train depth. {hybrid_order_status_sentence}

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(hybrid_order_comparison_rows, [
    ("mode", "mode"),
    ("train_max", "train max"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_formal_joint@16", "OOD formal joint"),
    ("ood_translated_joint@16", "OOD translated joint"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_formal_joint@16", "d50 formal joint"),
    ("depth50_translated_joint@16", "d50 translated joint"),
])}
\caption{{{hybrid_order_table_caption}}}
\end{{table}}
{hybrid_order_figure_block}

Sample inspection of the completed train-1-to-20/25 hybrid rows matches the intended surfaces: \texttt{{formal\_think}} starts with a \texttt{{<formal>}} block followed by \texttt{{<think>}}/\texttt{{<answer>}}, and \texttt{{think\_formal}} reverses that order. Shallow and train-band successes often have correct answers and translated NL proofs, but the formal block frequently omits explicit proof citations, so strict grounded formal validity is much lower than citation-free formal validity. At depth 50, both orders often truncate, lose required tags, or drift into long premise copying. The hybrid result should therefore be read as a sequence-format/interference result, not as evidence that combining both substrates gives more robust long-depth reasoning.

Conditioned dual-modality is different from the hybrid-order setup: the SFT data builder duplicates each materialized row into two separate examples, one prompted with \texttt{{<reasoning\_mode>formal\_logic</reasoning\_mode>}} and one prompted with \texttt{{<reasoning\_mode>natural\_language</reasoning\_mode>}}. Evaluation uses the same mode prompt. Thus there is no direct train-test mismatch where training has both traces in one target but eval asks for only one. The main caveat is exposure/interference: at fixed optimizer steps, each modality receives roughly half the updates of a single-modality run, and the shared adapter must fit both surfaces.

\begin{{table}}[H]
\centering
\scriptsize
{latex_table(conditioned_comparison_rows, [
    ("train_max", "train max"),
    ("condition", "condition"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
], max_rows=20)}
\caption{{Conditioned dual-modality ablation compared with the main single-modality baselines at each train-depth range.}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/ablation_conditioned_dual_vs_main_by_train_depth.pdf}}
\caption{{Conditioned dual-modality ablation compared with main logic and main NL across train-depth ranges. Dashed lines are conditioned runs; solid lines are single-modality baselines.}}
\end{{figure}}
\begin{{table}}[H]
\centering
\scriptsize
{latex_table(conditioned_50k_rows, [
    ("train_max", "train max"),
    ("eval_template", "eval template"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Conditioned dual-modality 50k final eval summary. {conditioned_50k_status_sentence}}}
\end{{table}}
{conditioned_50k_curve_block}

The completed 50k checkpoint samples preserve the intended conditioned prompts: formal-mode generations use \texttt{{<formal>}}/\texttt{{<answer>}}, while NL-mode generations use \texttt{{<think>}}/\texttt{{<answer>}}. The checkpoint curves therefore do not indicate a prompt mismatch. Instead, representative hard-depth samples show underexposure/interference symptoms: conditioned logic remains brittle on long proofs despite extra steps, and conditioned NL can produce valid translated train-band traces while depth-50 traces truncate or collapse into unsupported state lists. The pending final recovery rows are needed for the final 50k table, but the completed checkpoint sweep already argues against simple ``just train longer'' as a full explanation.

\section{{Qualitative samples}}
The companion PDF \texttt{{figures/sample\_generation\_panels.pdf}} contains synthetic generated examples with extracted answer, gold answer, correctness, and validity metadata. The following samples are from the completed bare-format OOD rerun for OLMo-7B train-1-to-25 seed 3407.

{latex_ood_examples(cot_bare_examples)}

\section{{Artifacts}}
{build_artifact_index_block()}

CSV tables live under \texttt{{tables/}} and all plots are emitted as both PDF and PNG under \texttt{{figures/}}. This machine currently lacks \texttt{{pdflatex}}/\texttt{{latexmk}}, so the LaTeX source is generated but not compiled here.
\end{{document}}
"""
    supplemental_figures = build_supplemental_figures_block(tex)
    if supplemental_figures:
        tex = tex.replace("\n\\section{Artifacts}\n", f"\n{supplemental_figures}\n\n\\section{{Artifacts}}\n")
    (OUT_ROOT / "logic_cot_report_2026-05-25.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    main_records, main_ckpt, tiny_records, tiny_ckpt, tiny_100k_records, tiny_100k_ckpt, qwen_records = load_records()
    symbol_padded_records = load_symbol_padded_records()
    wordified_records = load_wordified_records()
    extra_architecture_records = load_extra_architecture_records()
    architecture_records = main_records + qwen_records + extra_architecture_records
    all_summary = summarize_group(main_records + tiny_records + tiny_100k_records + qwen_records, (8, 16))
    main_summary = summarize_group(main_records, (8, 16))
    tiny_summary = summarize_group(tiny_records, (8,))
    tiny_100k_summary = summarize_group(tiny_100k_records, (8,))
    qwen_summary = summarize_group(qwen_records, (16,))
    architecture_summary = summarize_architecture(architecture_records)
    main_ood_summary = load_main_ood_summary()
    tiny_ood_summary = load_tiny_ood_summary()
    main_cot_bare_ood_summary = load_main_cot_bare_ood_summary()
    tiny_cot_bare_ood_summary = load_tiny_cot_bare_ood_summary(
        "ood_tiny_llama_cot_bare_2026-05-27",
        TINY_OOD_RE,
        "tiny_llama_cot_bare_ood",
    )
    tiny_100k_cot_bare_ood_summary = load_tiny_cot_bare_ood_summary(
        "ood_tiny_llama_100k_cot_bare_2026-05-27",
        TINY_100K_OOD_RE,
        "tiny_llama_100k_cot_bare_ood",
    )
    olmo32_cot_bare_gsm8k = load_olmo32_cot_bare_gsm8k()
    ablation_summaries = load_ablation_summaries()
    paired_full_summary = summarize_paired_full_suite()
    paired_igsm_semantic_summary = summarize_paired_igsm_semantic()
    active_paired_partial_summary = summarize_active_paired_partials()
    batch_size_partial_summary = summarize_hfsa_batch_size_partials()
    conditioned_50k_checkpoint_summary = load_conditioned_50k_checkpoint_summary()

    write_csv(TABLE_DIR / "main_olmo7b_summary.csv", main_summary.to_dict("records"))
    write_csv(TABLE_DIR / "tiny_llama_final_summary.csv", tiny_summary.to_dict("records"))
    write_csv(TABLE_DIR / "tiny_llama_100k_final_summary.csv", tiny_100k_summary.to_dict("records"))
    write_csv(TABLE_DIR / "qwen7b_partial_summary.csv", qwen_summary.to_dict("records"))
    write_csv(TABLE_DIR / "architecture_ablation_summary.csv", architecture_summary.to_dict("records"))
    write_csv(TABLE_DIR / "all_group_summary.csv", all_summary.to_dict("records"))
    write_csv(TABLE_DIR / "main_olmo7b_ood_lmeval_summary.csv", main_ood_summary.to_dict("records"))
    write_csv(TABLE_DIR / "tiny_llama_ood_lmeval_summary.csv", tiny_ood_summary.to_dict("records"))

    main_depth = depth_dataframe(main_records, DEPTHS_FINAL, 16)
    tiny_depth = depth_dataframe(tiny_records, DEPTHS_TINY, 8)
    main_depth.to_csv(TABLE_DIR / "main_olmo7b_depth_curves_k16.csv", index=False, lineterminator="\n")
    tiny_depth.to_csv(TABLE_DIR / "tiny_llama_depth_curves_k8.csv", index=False, lineterminator="\n")

    plot_main_trainmax(main_summary)
    plot_depth_grid(main_depth, "olmo7b_depth_correct16", "correct", 16)
    plot_depth_grid(main_depth, "olmo7b_depth_joint16", "joint", 16)
    plot_symbol_padded_depth_comparison(main_records, symbol_padded_records, wordified_records)
    plot_main_checkpoint_curves(main_ckpt)
    plot_main_checkpoint_depth_bands(main_ckpt)
    plot_tiny_final(tiny_records)
    plot_tiny_checkpoint_curves(tiny_records, tiny_ckpt)
    plot_tiny_checkpoint_depth_bands(tiny_records, tiny_ckpt, prefix="tiny_llama")
    plot_tiny_checkpoint_depth_bands(tiny_100k_records, tiny_100k_ckpt, prefix="tiny_llama_100k")
    plot_qwen_partial(qwen_records)
    plot_architecture_comparison(architecture_summary)
    plot_token_budget_comparison(main_summary, ablation_summaries.get("token_budget", pd.DataFrame()))
    plot_shortcut_comparison(main_summary, ablation_summaries.get("shortcut", pd.DataFrame()))
    plot_shortcut_kind_summary(ablation_summaries.get("shortcut_kind", pd.DataFrame()))
    plot_shortcut_kind_lines(main_summary, ablation_summaries.get("shortcut_kind", pd.DataFrame()))
    plot_trace_control_summary(build_trace_control_with_baselines(main_summary, ablation_summaries.get("trace_control", pd.DataFrame())))
    plot_hybrid_order_summary(build_hybrid_order_with_baselines(main_summary, ablation_summaries.get("hybrid_order", pd.DataFrame())))
    plot_paired_full_suite_partial(paired_full_summary)
    plot_paired_full_suite_family_partial(paired_full_summary)
    plot_paired_igsm_semantic(paired_igsm_semantic_summary)
    plot_active_paired_partials(active_paired_partial_summary)
    plot_hfsa_batch_size_partials(batch_size_partial_summary)
    plot_hfsa_batch_size_conditioned_deltas(batch_size_partial_summary)
    conditioned_comparison = build_conditioned_comparison_table(
        main_summary, ablation_summaries.get("conditioned", pd.DataFrame())
    )
    plot_conditioned_comparison(conditioned_comparison)
    plot_conditioned_50k_convergence(conditioned_50k_checkpoint_summary)

    samples = build_sample_panels()
    plot_sample_panels(samples)
    cot_bare_examples = build_cot_bare_generation_examples()
    write_report(
        main_summary,
        tiny_summary,
        tiny_100k_summary,
        qwen_summary,
        architecture_summary,
        main_ood_summary,
        tiny_ood_summary,
        main_cot_bare_ood_summary,
        tiny_cot_bare_ood_summary,
        tiny_100k_cot_bare_ood_summary,
        olmo32_cot_bare_gsm8k,
        ablation_summaries,
        paired_full_summary,
        paired_igsm_semantic_summary,
        active_paired_partial_summary,
        batch_size_partial_summary,
        cot_bare_examples,
        main_checkpoint_note(main_ckpt),
        len(tiny_ckpt),
        len(tiny_100k_ckpt),
    )

    print(f"wrote report artifacts to {OUT_ROOT}")
    print(f"main records: {len(main_records)}, main checkpoints: {len(main_ckpt)}")
    print(f"main OOD rows: {int(main_ood_summary['n'].sum()) if not main_ood_summary.empty else 0}")
    print(f"main bare OOD rows: {int(main_cot_bare_ood_summary['n'].sum()) if not main_cot_bare_ood_summary.empty else 0}")
    print(
        f"tiny final: {len(tiny_records)}, tiny checkpoints: {len(tiny_ckpt)}, "
        f"tiny 100k final: {len(tiny_100k_records)}, tiny 100k checkpoints: {len(tiny_100k_ckpt)}, "
        f"qwen: {len(qwen_records)}, architecture extras: {len(extra_architecture_records)}"
    )


if __name__ == "__main__":
    main()
