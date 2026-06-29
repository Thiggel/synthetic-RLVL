#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export generated synthetic proof traces as token-budgeted JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--template", choices=[TemplateName.LOGIC.value, TemplateName.NL_EXACT.value], required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--train-min-step", type=int, default=1)
    parser.add_argument("--train-max-step", type=int, default=25)
    parser.add_argument("--difficulty", default="hard_fsa_schema")
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--distractor-ratio", type=float, default=0.0)
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
        next_index = int(args.start_index)
        mode = "w"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = TaskConfig(
        template=TemplateName(args.template),
        prefill=PrefillMode.NONE,
        distractor_ratio=float(args.distractor_ratio),
        train_steps=StepRange(int(args.train_min_step), int(args.train_max_step)),
        val_steps=StepRange(int(args.train_min_step), int(args.train_max_step)),
        seed=int(args.seed),
        difficulty=str(args.difficulty),
        branching_factor=int(args.branching_factor),
    )
    builder = TaskBuilder(cfg)
    started = monotonic()

    def make_state(done: bool = False) -> dict:
        return {
            "kind": "generated_proof",
            "template": args.template,
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
        while tokens < int(args.target_tokens):
            if args.max_records is not None and records >= int(args.max_records):
                break
            sample = builder.sample(next_index, train=True)
            text = sample.prompt + sample.target
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            payload = {
                "text": text,
                "source": f"generated_hfsa_{args.template}",
                "template": args.template,
                "difficulty": args.difficulty,
                "depth": int(sample.depth),
                "record_index": int(next_index),
                "answer": sample.answer,
                "token_count": int(n_tokens),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            records += 1
            tokens += int(n_tokens)
            next_index += 1
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
                            "examples_per_second": records / elapsed,
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
