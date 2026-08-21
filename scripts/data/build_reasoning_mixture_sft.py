#!/usr/bin/env python3
"""Build Dolci + synthetic-trace SFT mixtures for the depth-threshold transfer test.

Design: REPLACEMENT, not addition. Every condition has exactly --total train
examples; conditions differ only in what the replaced slice contains. So
  control      = total Dolci
  <tmpl>-<band> = (total - k) Dolci + k synthetic traces, k = frac * total
This keeps example count and roughly the token budget constant, so a difference
cannot be attributed to simply training on more data.

The eval split is left Dolci-only in every condition so eval loss stays
comparable across conditions.

Traces are rendered through the SAME path as the controlled BranchProof grid
(task_sample_from_materialized_row), so the trace text is identical to the
study that established the depth crossover.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk

from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName

BANDS = {15: "train_fixedtarget_up_to_15_50k", 25: "train_fixedtarget_up_to_25_50k"}


def render_traces(bp_root: Path, band: int, template: str, k: int, seed: int) -> list[dict]:
    table = pq.read_table(bp_root / BANDS[band] / "train.parquet")
    cols = table.column_names
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(table.num_rows), k))
    cfg = TaskConfig(
        template=TemplateName(template),
        prefill=PrefillMode.NONE,
        distractor_ratio=0.0,
        train_steps=StepRange(1, band),
        val_steps=StepRange(1, band),
        seed=seed,
    )
    out = []
    for i in idx:
        row = {c: table.column(c)[i].as_py() for c in cols}
        sample = task_sample_from_materialized_row(row, cfg=cfg)
        out.append({"prompt": sample.prompt, "target": sample.target})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolci", type=Path, required=True)
    ap.add_argument("--bp-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--band", type=int, required=True, choices=sorted(BANDS))
    ap.add_argument("--template", required=True, choices=["logic", "nl_exact"])
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    dolci = load_from_disk(str(args.dolci))
    k = int(round(args.frac * args.total))
    n_dolci = args.total - k
    if len(dolci["train"]) < args.total:
        raise SystemExit(f"dolci train has {len(dolci['train'])} < total {args.total}")

    kept = dolci["train"].select(range(n_dolci))
    traces = render_traces(args.bp_root, args.band, args.template, k, args.seed)

    rows = [{"prompt": p, "target": t} for p, t in zip(kept["prompt"], kept["target"])] + traces
    random.Random(args.seed).shuffle(rows)

    out_dir = args.out_root / f"dolci_bp_{args.template}_band{args.band}_p{int(args.frac*100)}"
    DatasetDict({"train": Dataset.from_list(rows), "eval": dolci["eval"]}).save_to_disk(str(out_dir))

    meta = {
        "condition": f"{args.template}_band{args.band}",
        "template": args.template,
        "band": args.band,
        "frac": args.frac,
        "total": args.total,
        "n_dolci": n_dolci,
        "n_traces": k,
        "seed": args.seed,
        "dolci_source": str(args.dolci),
        "bp_source": str(args.bp_root / BANDS[args.band]),
        "design": "replacement (constant example count); eval split is Dolci-only",
    }
    (out_dir / "mixture_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
