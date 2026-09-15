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
    args = ap.parse_args()

    raw = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    rows, sources = [], {}
    need = args.train + args.eval
    for item in raw:
        messages = item.get("messages") or []
        if len(messages) != 2:
            continue
        user, assistant = messages
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            continue
        prompt, target = user.get("content", ""), assistant.get("content", "")
        if not prompt or not target:
            continue
        if len(prompt) + len(target) > args.max_chars:
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
    meta = dict(dataset=args.dataset, train=args.train, eval=args.eval,
                seed=args.seed, max_chars=args.max_chars,
                sources=dict(sorted(sources.items(), key=lambda kv: -kv[1])[:15]))
    (args.out / "corpus_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
