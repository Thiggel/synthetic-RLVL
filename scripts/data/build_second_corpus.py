#!/usr/bin/env python3
"""Prepare a second instruction corpus in the format the trainer expects.

The paper claims that instruction tuning installs the affirmative answer prior
and that the generated derivations correct it. That claim rests on one
instruction corpus, so it has to be repeated on another one drawn from a
different source with a different mixture of tasks.

Single-turn examples only, so the format matches the first corpus exactly and
the comparison isolates the corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=int, default=100000)
    ap.add_argument("--eval", type=int, default=2048)
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B",
                    help="used only to drop examples the chat template cannot round-trip")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def round_trips(prompt: str, target: str) -> bool:
        """The trainer masks the prompt by prefix, so an example is only usable
        when the full rendering starts with the prompt rendering."""
        try:
            head = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=True, add_generation_prompt=True)
            full = tok.apply_chat_template([{"role": "user", "content": prompt},
                                            {"role": "assistant", "content": target}],
                                           tokenize=True)
        except Exception:
            return False
        return len(full) > len(head) and full[: len(head)] == head

    raw = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    rows, sources, dropped = [], {}, [0]
    need = args.train + args.eval
    for item in raw:
        messages = item.get("messages") or []
        if len(messages) != 2:
            continue
        user, assistant = messages
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            continue
        prompt = (user.get("content") or "").strip()
        target = (assistant.get("content") or "").strip()
        if not prompt or not target:
            continue
        if len(prompt) + len(target) > args.max_chars:
            continue
        if not round_trips(prompt, target):
            dropped[0] += 1
            continue
        rows.append({"prompt": prompt, "target": target})
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        if len(rows) >= need:
            break
    if len(rows) < need:
        raise SystemExit(f"only {len(rows)} single-turn examples, need {need}")

    args.out.mkdir(parents=True, exist_ok=True)
    DatasetDict({
        "train": Dataset.from_list(rows[: args.train]),
        "eval": Dataset.from_list(rows[args.train:need]),
    }).save_to_disk(str(args.out))
    meta = dict(dataset=args.dataset, train=args.train, eval=args.eval, dropped=dropped[0],
                seed=args.seed, max_chars=args.max_chars,
                sources=dict(sorted(sources.items(), key=lambda kv: -kv[1])[:15]))
    (args.out / "corpus_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
