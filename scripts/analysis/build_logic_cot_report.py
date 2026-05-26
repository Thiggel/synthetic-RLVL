from __future__ import annotations

import csv
import json
import os
import re
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
WORK_ROOT = Path(os.environ.get("WORK", ROOT))
PASSK_ROOT = WORK_ROOT / "synthetic-RLVL" / "passk_eval"
LM_EVAL_ROOT = WORK_ROOT / "synthetic-RLVL" / "lm_eval_results"
OUT_ROOT = ROOT / "analysis" / "logic_cot_report_2026-05-25"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

MAIN_RE = re.compile(r"sft_hfsa_depth_scaling_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$")
MAIN_CKPT_RE = re.compile(
    r"sft_hfsa_depth_scaling_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_checkpoint-(\d+)_passk\.json$"
)
TINY_RE = re.compile(r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_seed(\d+)_passk\.json$")
TINY_CKPT_RE = re.compile(
    r"pretrain_hfsa_llama3_(50m|100m|200m)_(logic|nl_exact)_train1to10_seed(\d+)_checkpoint-(\d+)_passk\.json$"
)
QWEN_RE = re.compile(
    r"sft_hfsa_modelablate_qwen2p5_7b_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)_passk\.json$"
)
MAIN_OOD_RE = re.compile(r"sft_hfsa_depth_scaling_(logic|nl_exact)_train1to(\d+)_10k_seed(\d+)$")

DEPTHS_FINAL = [1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
DEPTHS_TINY = [1, 2, 5, 10, 12, 15, 18, 20, 25, 30, 40, 50]
TEMPLATE_LABEL = {"logic": "Logic", "nl_exact": "NL exact"}
COLORS = {"logic": "#1f77b4", "nl_exact": "#d62728"}
SIZE_ORDER = {"50m": 50, "100m": 100, "200m": 200}


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
    return "citation_free_joint_pass" if template == "logic" else "nl_logic_joint_pass"


def valid_metric(template: str) -> str:
    return "citation_free_valid_pass" if template == "logic" else "nl_logic_citation_free_valid_pass"


def read_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(record: Record, suffix: str) -> float | None:
    value = record.metrics.get(f"synthetic_sampled/{suffix}")
    return float(value) if isinstance(value, (int, float)) else None


def band_metric(record: Record, band: str, name: str, k: int) -> float | None:
    return metric(record, f"band_{band}/{name}@{k}")


def step_metric(record: Record, depth: int, name: str, k: int) -> float | None:
    return metric(record, f"step_{depth}/{name}@{k}")


def f3(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def avg_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return None, None, 0
    return mean(clean), pstdev(clean) if len(clean) > 1 else 0.0, len(clean)


def load_records() -> tuple[list[Record], list[Record], list[Record], list[Record], list[Record]]:
    main: list[Record] = []
    main_ckpt: list[Record] = []
    tiny: list[Record] = []
    tiny_ckpt: list[Record] = []
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
    for path in sorted((PASSK_ROOT / "hfsa_model_ablation_qwen2p5_7b_sparse").glob("*_passk.json")):
        if match := QWEN_RE.match(path.name):
            payload = read_payload(path)
            qwen.append(
                Record("qwen7b", path, match.group(1), int(match.group(2)), int(match.group(3)), payload["metrics"])
            )
    return main, main_ckpt, tiny, tiny_ckpt, qwen


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
    integer_columns = {"n", "seed", "train_max", "checkpoint", "depth"}
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
        rows.append(
            {
                "template": match.group(1),
                "train_max": int(match.group(2)),
                "seed": int(match.group(3)),
                "gsm8k_em": get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "exact_match,none"),
                "gsm8k_tag": get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "tag_found,none"),
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

        rows.append(
            {
                "size": match.group(1),
                "template": match.group(2),
                "seed": int(match.group(3)),
                "gsm8k_em": get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "exact_match,none"),
                "gsm8k_tag": get_lm_eval_metric(results, "synthrlvl_gsm8k_tagged", "tag_found,none"),
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


def write_report(
    main_summary: pd.DataFrame,
    tiny_summary: pd.DataFrame,
    qwen_summary: pd.DataFrame,
    main_ood_summary: pd.DataFrame,
    tiny_ood_summary: pd.DataFrame,
    main_ckpt_note: str,
    tiny_ckpt_count: int,
) -> None:
    main_rows = main_summary[(main_summary["size"] == "") & (main_summary["n"] >= 1)].copy()
    main_rows = main_rows.sort_values(["template", "train_max"])
    main_ood_rows = main_ood_summary.sort_values(["template", "train_max"]) if not main_ood_summary.empty else pd.DataFrame()
    tiny_rows = tiny_summary[tiny_summary["size"] != ""].copy() if not tiny_summary.empty else pd.DataFrame()
    if not tiny_rows.empty:
        tiny_rows["size_order"] = tiny_rows["size"].map(SIZE_ORDER)
        tiny_rows = tiny_rows.sort_values(["size_order", "template"])
    qwen_rows = qwen_summary.sort_values(["template", "train_max"]) if not qwen_summary.empty else pd.DataFrame()
    train_maxes = [5, 10, 15, 20, 25]
    tiny_sizes = ["50m", "100m", "200m"]
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

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\title{{Formal Logic CoT Synthetic Results Update}}
\date{{2026-05-26}}
\begin{{document}}
\maketitle

\section{{Metric note for OOD benchmarks}}
EM means exact match: after extracting the model's answer, the evaluator normalizes the prediction and gold answer and assigns 1 only when they exactly match, then averages over examples. For GSM8K this is numeric exact match and is effectively an accuracy over single numeric answers. For HotpotQA, 2WikiMultiHopQA, and MuSiQue the answers are free-form strings with aliases and partial-overlap possibilities, so the standard report is EM plus token-level F1 rather than calling the result plain accuracy. F1 gives partial credit for overlapping answer tokens; EM is the stricter all-or-nothing string match.

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

\section{{Qwen-2.5-7B partial architecture ablation}}
\begin{{table}}[H]
\centering
\small
{latex_table(qwen_rows, [
    ("template", "template"),
    ("train_max", "train max"),
    ("n", "n"),
    ("ood_correct@16", "OOD c@16"),
    ("ood_joint@16", "OOD joint@16"),
    ("depth50_correct@16", "d50 c@16"),
    ("depth50_joint@16", "d50 joint@16"),
])}
\caption{{Qwen-2.5-7B sparse eval rows available at report time. Matched NL train-1-to-20/25 rows are still incomplete, so this is not yet a full architecture-ablation conclusion.}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.86\linewidth]{{figures/qwen7b_partial_ood_correct_joint.pdf}}
\caption{{Partial Qwen-2.5-7B OOD correct/joint@16.}}
\end{{figure}}

\section{{Qualitative samples}}
The companion PDF \texttt{{figures/sample\_generation\_panels.pdf}} contains generated examples with extracted answer, gold answer, correctness, and validity metadata. This is kept as a PDF figure rather than long verbatim text in the main report.

\section{{Artifacts}}
CSV tables live under \texttt{{tables/}} and all plots are emitted as both PDF and PNG under \texttt{{figures/}}. This machine currently lacks \texttt{{pdflatex}}/\texttt{{latexmk}}, so the LaTeX source is generated but not compiled here.
\end{{document}}
"""
    (OUT_ROOT / "logic_cot_report_2026-05-25.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    main_records, main_ckpt, tiny_records, tiny_ckpt, qwen_records = load_records()
    all_summary = summarize_group(main_records + tiny_records + qwen_records, (8, 16))
    main_summary = summarize_group(main_records, (8, 16))
    tiny_summary = summarize_group(tiny_records, (8,))
    qwen_summary = summarize_group(qwen_records, (16,))
    main_ood_summary = load_main_ood_summary()
    tiny_ood_summary = load_tiny_ood_summary()

    write_csv(TABLE_DIR / "main_olmo7b_summary.csv", main_summary.to_dict("records"))
    write_csv(TABLE_DIR / "tiny_llama_final_summary.csv", tiny_summary.to_dict("records"))
    write_csv(TABLE_DIR / "qwen7b_partial_summary.csv", qwen_summary.to_dict("records"))
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
    plot_main_checkpoint_curves(main_ckpt)
    plot_tiny_final(tiny_records)
    plot_tiny_checkpoint_curves(tiny_records, tiny_ckpt)
    plot_qwen_partial(qwen_records)

    samples = build_sample_panels()
    plot_sample_panels(samples)
    write_report(
        main_summary,
        tiny_summary,
        qwen_summary,
        main_ood_summary,
        tiny_ood_summary,
        main_checkpoint_note(main_ckpt),
        len(tiny_ckpt),
    )

    print(f"wrote report artifacts to {OUT_ROOT}")
    print(f"main records: {len(main_records)}, main checkpoints: {len(main_ckpt)}")
    print(f"main OOD rows: {int(main_ood_summary['n'].sum()) if not main_ood_summary.empty else 0}")
    print(f"tiny final: {len(tiny_records)}, tiny checkpoints: {len(tiny_ckpt)}, qwen: {len(qwen_records)}")


if __name__ == "__main__":
    main()
