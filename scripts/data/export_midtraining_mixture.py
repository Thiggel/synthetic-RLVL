#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic_dataset import MaterializedSyntheticDataset
from synthrlvl.task import task_sample_from_materialized_row
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: str
    weight: int = 1
    dataset_id: str | None = None
    local_root: str | None = None
    subset: str | None = None
    template: str = "logic"
    path: str | None = None
    text_field: str = "text"
    limit: int | None = None


def _parse_component(raw: str) -> ComponentSpec:
    parts: dict[str, str] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid component chunk {chunk!r}; expected key=value")
        key, value = chunk.split("=", 1)
        parts[key.strip()] = value.strip()
    if "name" not in parts:
        raise ValueError(f"Component missing name: {raw}")
    kind = parts.get("kind", "materialized")
    return ComponentSpec(
        name=parts["name"],
        kind=kind,
        weight=max(1, int(parts.get("weight", "1"))),
        dataset_id=parts.get("dataset_id"),
        local_root=parts.get("local_root"),
        subset=parts.get("subset"),
        template=parts.get("template", "logic"),
        path=parts.get("path"),
        text_field=parts.get("text_field", "text"),
        limit=int(parts["limit"]) if "limit" in parts and parts["limit"] else None,
    )


def _task_cfg(template: str) -> TaskConfig:
    return TaskConfig(
        template=TemplateName(template),
        prefill=PrefillMode.NONE,
        distractor_ratio=0.0,
        train_steps=StepRange(1, 1),
        val_steps=StepRange(1, 1),
        seed=0,
        difficulty="hard_fsa_schema",
        branching_factor=4,
    )


def _iter_materialized(spec: ComponentSpec, *, output_mode: str) -> Iterator[dict]:
    if not spec.subset:
        raise ValueError(f"Materialized component {spec.name!r} requires subset=...")
    ds = MaterializedSyntheticDataset()
    rows = ds.load_rows(
        subset=spec.subset,
        dataset_id=spec.dataset_id,
        local_root=spec.local_root,
        split="train",
        limit=spec.limit,
    )
    cfg = _task_cfg(spec.template)
    for row in rows:
        sample = task_sample_from_materialized_row(row, cfg=cfg)
        if output_mode == "causal":
            text = sample.prompt + sample.target
            payload = {"text": text}
        elif output_mode == "sft":
            payload = {"prompt": sample.prompt, "completion": sample.target, "text": sample.prompt + sample.target}
        else:
            raise ValueError(f"Unsupported output mode: {output_mode}")
        payload.update(
            {
                "source": spec.name,
                "kind": spec.kind,
                "template": spec.template,
                "subset": spec.subset,
                "depth": int(row.get("depth", sample.depth)),
                "record_index": int(row.get("record_index", -1)),
                "answer": sample.answer,
            }
        )
        yield payload


def _iter_jsonl(spec: ComponentSpec) -> Iterator[dict]:
    if not spec.path:
        raise ValueError(f"JSONL component {spec.name!r} requires path=...")
    count = 0
    with Path(spec.path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if spec.limit is not None and count >= spec.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get(spec.text_field, ""))
            if not text:
                continue
            row.setdefault("text", text)
            row.setdefault("source", spec.name)
            row.setdefault("kind", spec.kind)
            count += 1
            yield row


def _iter_component(spec: ComponentSpec, *, output_mode: str) -> Iterator[dict]:
    if spec.kind == "materialized":
        return _iter_materialized(spec, output_mode=output_mode)
    if spec.kind == "jsonl":
        return _iter_jsonl(spec)
    raise ValueError(f"Unsupported component kind {spec.kind!r}")


def _token_counter(tokenizer_name: str | None):
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    return lambda text: len(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export token-budgeted midtraining JSONL mixtures.")
    parser.add_argument("--component", action="append", required=True, help="Comma-separated key=value component spec.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-mode", choices=["causal", "sft"], default="causal")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer for token-budget accounting.")
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = [_parse_component(raw) for raw in args.component]
    counter = _token_counter(args.tokenizer)
    iters = {spec.name: _iter_component(spec, output_mode=args.output_mode) for spec in specs}
    schedule = list(itertools.chain.from_iterable([[spec.name] * spec.weight for spec in specs]))
    if not schedule:
        raise ValueError("Empty mixture schedule.")

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "components": [spec.__dict__ for spec in specs],
        "output_mode": args.output_mode,
        "tokenizer": args.tokenizer,
        "max_total_tokens": args.max_total_tokens,
        "max_records": args.max_records,
        "records": 0,
        "tokens": 0,
        "by_source": {},
    }

    exhausted: set[str] = set()
    records = 0
    tokens = 0
    with out.open("w", encoding="utf-8") as handle:
        while len(exhausted) < len(iters):
            progressed = False
            for name in schedule:
                if name in exhausted:
                    continue
                if args.max_records is not None and records >= args.max_records:
                    exhausted.update(iters)
                    break
                try:
                    row = next(iters[name])
                except StopIteration:
                    exhausted.add(name)
                    continue
                text = str(row.get("text", ""))
                n_tokens = int(counter(text))
                if args.max_total_tokens is not None and tokens + n_tokens > args.max_total_tokens:
                    exhausted.update(iters)
                    break
                row["token_count"] = n_tokens
                tokens += n_tokens
                records += 1
                src = str(row.get("source", name))
                stats = manifest["by_source"].setdefault(src, {"records": 0, "tokens": 0})
                stats["records"] += 1
                stats["tokens"] += n_tokens
                if not args.dry_run:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                progressed = True
            if not progressed:
                break

    manifest["records"] = records
    manifest["tokens"] = tokens
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
