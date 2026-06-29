#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a Hugging Face text dataset into a token-budgeted JSONL file.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(out.suffix + ".state.json")
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")

    state = _load_state(state_path) if args.resume else None
    if state:
        if not out.exists():
            raise FileNotFoundError(f"Resume state exists but output JSONL is missing: {out}")
        byte_offset = int(state.get("byte_offset", out.stat().st_size))
        with out.open("rb+") as handle:
            handle.truncate(byte_offset)
        records = int(state["records"])
        tokens = int(state["tokens"])
        next_index = int(state["next_index"])
        mode = "a"
    else:
        if out.exists():
            raise FileExistsError(f"{out} exists without usable resume state; remove it or restore {state_path}.")
        byte_offset = 0
        records = 0
        tokens = 0
        next_index = 0
        mode = "w"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.config:
        stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    else:
        stream = load_dataset(args.dataset, split=args.split, streaming=True)
    if next_index:
        stream = stream.skip(next_index)
    started = monotonic()

    def make_state(done: bool = False) -> dict:
        return {
            "kind": "hf_text",
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "text_field": args.text_field,
            "tokenizer": args.tokenizer,
            "target_tokens": int(args.target_tokens),
            "records": int(records),
            "tokens": int(tokens),
            "next_index": int(next_index),
            "byte_offset": int(byte_offset),
            "done": bool(done),
            "output": str(out),
        }

    with out.open(mode, encoding="utf-8") as handle:
        for row in stream:
            if tokens >= int(args.target_tokens):
                break
            if args.max_records is not None and records >= int(args.max_records):
                break
            text = str(row.get(args.text_field, "")).strip()
            next_index += 1
            if not text:
                continue
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            payload = {
                "text": text,
                "source": args.dataset,
                "dataset_config": args.config,
                "split": args.split,
                "record_index": int(next_index - 1),
                "token_count": int(n_tokens),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            records += 1
            tokens += int(n_tokens)
            if records % int(args.state_every) == 0:
                handle.flush()
                byte_offset = handle.buffer.tell()
                _write_json(state_path, make_state(done=False))
            if records % int(args.progress_every) == 0:
                elapsed = max(monotonic() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "records": records,
                            "tokens": tokens,
                            "target_tokens": int(args.target_tokens),
                            "records_per_second": records / elapsed,
                        }
                    ),
                    flush=True,
                )

    with out.open("ab") as handle:
        byte_offset = handle.tell()
    final_state = make_state(done=tokens >= int(args.target_tokens))
    _write_json(state_path, final_state)
    _write_json(manifest_path, final_state)
    print(json.dumps(final_state, indent=2), flush=True)


if __name__ == "__main__":
    main()
