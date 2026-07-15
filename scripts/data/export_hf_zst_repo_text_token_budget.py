#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from pathlib import Path
from time import monotonic

import zstandard
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export text directly from shuffled JSONL.zst shards on the HF Hub.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--file-prefix", default="data/")
    parser.add_argument("--shuffle-seed", type=int, default=42)
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


def _iter_zst_jsonl(path: Path, skip_lines: int = 0):
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
                for line_index, line in enumerate(text_reader):
                    if line_index < skip_lines or not line.strip():
                        continue
                    yield line_index, json.loads(line)


def main() -> None:
    args = parse_args()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(out.suffix + ".state.json")
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")

    files = [
        name
        for name in HfApi().list_repo_files(args.dataset, repo_type="dataset")
        if name.startswith(args.file_prefix) and name.endswith(".jsonl.zst")
    ]
    if not files:
        raise RuntimeError(f"No JSONL.zst files found under {args.dataset}:{args.file_prefix}")
    files.sort()
    random.Random(int(args.shuffle_seed)).shuffle(files)

    state = _load_state(state_path) if args.resume else None
    if state:
        expected = {
            "dataset": args.dataset,
            "tokenizer": args.tokenizer,
            "target_tokens": int(args.target_tokens),
            "file_prefix": args.file_prefix,
            "shuffle_seed": int(args.shuffle_seed),
            "num_repo_files": len(files),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Resume state mismatch for {key}: {state.get(key)!r} != {value!r}")
        if not out.exists():
            raise FileNotFoundError(f"Resume state exists but output JSONL is missing: {out}")
        byte_offset = int(state.get("byte_offset", out.stat().st_size))
        with out.open("rb+") as handle:
            handle.truncate(byte_offset)
        records = int(state["records"])
        tokens = int(state["tokens"])
        file_position = int(state["file_position"])
        next_line_index = int(state["next_line_index"])
        source_records = Counter({str(k): int(v) for k, v in state.get("source_records", {}).items()})
        source_tokens = Counter({str(k): int(v) for k, v in state.get("source_tokens", {}).items()})
        mode = "a"
    else:
        if out.exists():
            raise FileExistsError(f"{out} exists without usable resume state; remove it or restore {state_path}.")
        byte_offset = 0
        records = 0
        tokens = 0
        file_position = 0
        next_line_index = 0
        source_records = Counter()
        source_tokens = Counter()
        mode = "w"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    started = monotonic()

    def make_state(done: bool = False) -> dict:
        return {
            "kind": "hf_zst_repo_text",
            "dataset": args.dataset,
            "tokenizer": args.tokenizer,
            "target_tokens": int(args.target_tokens),
            "file_prefix": args.file_prefix,
            "shuffle_seed": int(args.shuffle_seed),
            "num_repo_files": len(files),
            "records": int(records),
            "tokens": int(tokens),
            "file_position": int(file_position),
            "next_line_index": int(next_line_index),
            "source_records": dict(sorted(source_records.items())),
            "source_tokens": dict(sorted(source_tokens.items())),
            "byte_offset": int(byte_offset),
            "done": bool(done),
            "output": str(out),
        }

    with out.open(mode, encoding="utf-8") as handle:
        while file_position < len(files) and tokens < int(args.target_tokens):
            repo_file = files[file_position]
            local_path = Path(hf_hub_download(args.dataset, repo_file, repo_type="dataset"))
            for line_index, row in _iter_zst_jsonl(local_path, skip_lines=next_line_index):
                if tokens >= int(args.target_tokens):
                    break
                if args.max_records is not None and records >= int(args.max_records):
                    break
                next_line_index = line_index + 1
                text = str(row.get("text", ""))
                if not text.strip():
                    continue
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                source = repo_file.split("/", 2)[1] if "/" in repo_file else repo_file
                payload = {
                    "text": text,
                    "source": source,
                    "repo_file": repo_file,
                    "record_index": int(records),
                    "token_count": int(n_tokens),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                records += 1
                tokens += int(n_tokens)
                source_records[source] += 1
                source_tokens[source] += int(n_tokens)
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
                                "file_position": file_position,
                                "repo_file": repo_file,
                                "records_per_second": records / elapsed,
                            }
                        ),
                        flush=True,
                    )
            if args.max_records is not None and records >= int(args.max_records):
                break
            if tokens < int(args.target_tokens):
                file_position += 1
                next_line_index = 0
                handle.flush()
                byte_offset = handle.buffer.tell()
                _write_json(state_path, make_state(done=False))

    with out.open("ab") as handle:
        byte_offset = handle.tell()
    done = tokens >= int(args.target_tokens)
    final_state = make_state(done=done)
    _write_json(state_path, final_state)
    _write_json(manifest_path, final_state)
    print(json.dumps(final_state, indent=2), flush=True)
    if not done and args.max_records is None:
        raise RuntimeError(f"Exhausted {len(files)} shards at {tokens} tokens before target {args.target_tokens}")


if __name__ == "__main__":
    main()
