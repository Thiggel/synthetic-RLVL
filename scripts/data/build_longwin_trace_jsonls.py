#!/usr/bin/env python
"""Render band-25 midtrain trace corpora for the long-window (8192) midtrains.

Documents are the SFT-style rendering (task_sample_from_materialized_row,
prompt + target concatenated — the exact text train_instruction_sft.py trains
on), NOT the old compact midtrain rendering. Three renderings from the SAME
latent proofs:

- logic      (template=logic)
- nl_exact   (template=nl_exact)
- condensed_logic (scripts/data/condensed_formal_rendering.render_condensed)

Writes one JSONL per rendering ({"text", "depth"}) plus a token-length stats
JSON per rendering (p50/p95/p99/max, frac>4096, frac>8192) measured with the
actual training tokenizer. Fail-closed: any document longer than --window
aborts the build.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def stats(lens, window):
    import numpy as np

    arr = np.asarray(lens)
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "total_tokens": int(arr.sum()),
        "frac_gt_4096": float((arr > 4096).mean()),
        "frac_gt_window": float((arr > window).mean()),
        "window": window,
        "histogram_bin_edges": [int(x) for x in range(0, int(arr.max()) + 500, 250)],
        "histogram_counts": [int(c) for c in np.histogram(arr, bins=range(0, int(arr.max()) + 500, 250))[0]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--band", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260830, help="TaskConfig seed (rendering)")
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--renderings", default="logic,nl_exact,condensed_logic")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from condensed_formal_rendering import render_condensed
    from synthrlvl.task import task_sample_from_materialized_row
    from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    table = pq.read_table(args.parquet)
    cols = table.column_names
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    renderings = [r.strip() for r in args.renderings.split(",") if r.strip()]

    cfgs = {
        name: TaskConfig(
            template=TemplateName(name),
            prefill=PrefillMode.NONE,
            distractor_ratio=0.0,
            train_steps=StepRange(1, args.band),
            val_steps=StepRange(1, args.band),
            seed=args.seed,
        )
        for name in renderings
        if name != "condensed_logic"
    }

    handles = {name: (out_root / f"{name}_band{args.band}.jsonl").open("w", encoding="utf-8") for name in renderings}
    lens = {name: [] for name in renderings}
    batch_texts = {name: [] for name in renderings}

    def flush(name):
        if not batch_texts[name]:
            return
        enc = tok(batch_texts[name], add_special_tokens=False)["input_ids"]
        lens[name].extend(len(ids) for ids in enc)
        batch_texts[name] = []

    n_rows = table.num_rows
    for i in range(n_rows):
        row = {c: table.column(c)[i].as_py() for c in cols}
        depth = int(row["depth"])
        for name in renderings:
            if name == "condensed_logic":
                text = render_condensed(row)
            else:
                s = task_sample_from_materialized_row(row, cfg=cfgs[name])
                text = s.prompt + s.target
            handles[name].write(json.dumps({"text": text, "depth": depth}) + "\n")
            batch_texts[name].append(text)
            if len(batch_texts[name]) >= 512:
                flush(name)
        if (i + 1) % 5000 == 0:
            print(f"rendered {i + 1}/{n_rows}", flush=True)
    for name in renderings:
        flush(name)
        handles[name].close()

    manifest = {"parquet": args.parquet, "tokenizer": args.tokenizer, "band": args.band, "seed": args.seed, "rows": n_rows, "renderings": {}}
    failed = []
    for name in renderings:
        s = stats(lens[name], args.window)
        s["jsonl"] = str(out_root / f"{name}_band{args.band}.jsonl")
        manifest["renderings"][name] = s
        (out_root / f"{name}_band{args.band}.stats.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        if s["frac_gt_window"] > 0:
            failed.append(name)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("histogram")} for k, v in manifest["renderings"].items()}, indent=2))
    if failed:
        raise SystemExit(f"FAIL-CLOSED: renderings with documents over the {args.window} window: {failed}")


if __name__ == "__main__":
    main()
