#!/usr/bin/env python
"""Long-document control docpack (Arm 2 of the long-window midtrain).

Selects LONG Dolmino documents (no deep deductive content — ordinary Dolmino
text) from the already-tokenized Dolmino nanoset, length-matched to the
band-25 logic trace token-length histogram, and packs them with the
document-preserving packer into seq_len+1 windows.

Source: the packed Dolmino nanoset itself (.ds + .ds.index doc boundaries),
so token ids are exactly what training would see — no decode/re-tokenize
round-trip. Documents containing the padding token id are skipped (counted).

Length matching: the band-25 stats JSON (build_longwin_trace_jsonls.py)
provides a 250-token-bin histogram; Dolmino docs are drawn per bin
proportionally until --target-real-tokens is reached. The achieved histogram
and the availability per bin are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from pack_document_preserving_nanoset import (  # noqa: E402
    pack_documents,
    shuffle_windows,
    summarize,
    write_nanoset_folder,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dolmino-folder", required=True)
    ap.add_argument("--band25-stats-json", required=True, help="logic band-25 stats from build_longwin_trace_jsonls.py")
    ap.add_argument("--output-folder", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--pad-token", default="<|fim_pad|>")
    ap.add_argument("--target-real-tokens", type=int, default=262_000_000)
    ap.add_argument("--token-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--shuffle-seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    pad_id = tokenizer.convert_tokens_to_ids(args.pad_token)
    assert eos_id is not None and pad_id is not None and eos_id != pad_id

    dtype = {2: np.uint16, 4: np.uint32}[args.token_size]
    shards = sorted(Path(args.dolmino_folder).glob("*.ds"))
    if not shards:
        raise SystemExit(f"no .ds shards under {args.dolmino_folder}")

    band = json.loads(Path(args.band25_stats_json).read_text())
    edges = np.asarray(band["histogram_bin_edges"], dtype=np.int64)
    counts = np.asarray(band["histogram_counts"], dtype=np.float64)
    if counts.sum() <= 0:
        raise SystemExit("empty band-25 histogram")
    probs = counts / counts.sum()
    bin_lo, bin_hi = edges[:-1], edges[1:]

    # Collect (shard_idx, start, end) per doc, bucketed by token length.
    per_bin_docs = [[] for _ in range(len(probs))]
    n_docs_total = 0
    for s_i, shard in enumerate(shards):
        idx = np.fromfile(str(shard) + ".index", dtype=np.uint64).astype(np.int64)
        starts = np.concatenate([[0], idx[:-1]])
        lengths = idx - starts
        n_docs_total += len(idx)
        which = np.searchsorted(bin_lo, lengths, side="right") - 1
        ok = (which >= 0) & (which < len(probs)) & (lengths <= bin_hi[np.clip(which, 0, len(probs) - 1)])
        for d_i in np.nonzero(ok)[0]:
            b = int(which[d_i])
            if probs[b] > 0:
                per_bin_docs[b].append((s_i, int(starts[d_i]), int(idx[d_i])))

    rng = np.random.RandomState(args.seed)
    for b in range(len(probs)):
        rng.shuffle(per_bin_docs[b])

    availability = {
        f"{int(bin_lo[b])}-{int(bin_hi[b])}": {"available": len(per_bin_docs[b]), "band25_count": int(counts[b])}
        for b in range(len(probs))
        if counts[b] > 0
    }

    # Draw docs bin-by-bin proportionally until target tokens reached.
    maps = [np.memmap(sh, dtype=dtype, mode="r") for sh in shards]
    chosen: list[tuple[int, int, int]] = []
    total = 0
    cursors = [0] * len(probs)
    skipped_pad = 0
    exhausted = set()
    expected_tokens_per_draw = float(np.dot(probs, (bin_lo + bin_hi) / 2.0))
    while total < args.target_real_tokens:
        # Sample a bin according to the band-25 distribution.
        b = int(rng.choice(len(probs), p=probs))
        while cursors[b] >= len(per_bin_docs[b]):
            exhausted.add(b)
            b = int(rng.choice(len(probs), p=probs))
            if len(exhausted) == int((probs > 0).sum()):
                raise SystemExit(
                    f"Dolmino long-doc supply exhausted at {total} tokens "
                    f"(target {args.target_real_tokens}); PG19 fallback needed."
                )
        s_i, start, end = per_bin_docs[b][cursors[b]]
        cursors[b] += 1
        doc = np.asarray(maps[s_i][start:end], dtype=np.int64)
        if (doc == pad_id).any():
            skipped_pad += 1
            continue
        if doc[-1] != eos_id:
            doc = np.concatenate([doc, [eos_id]])
        chosen.append((s_i, start, end))
        total += len(doc)

    print(f"selected {len(chosen)} docs / {total} tokens (skipped {skipped_pad} pad-colliding)", flush=True)

    def doc_iter():
        for doc_id, (s_i, start, end) in enumerate(chosen):
            doc = np.asarray(maps[s_i][start:end], dtype=np.int64).tolist()
            if doc[-1] != eos_id:
                doc.append(eos_id)
            yield doc_id, doc

    window_len = args.seq_len + 1
    result = pack_documents(doc_iter(), window_len=window_len, pad_id=pad_id)
    result = shuffle_windows(result, seed=args.shuffle_seed)
    out_folder = Path(args.output_folder).expanduser().resolve()
    write_info = write_nanoset_folder(result, out_folder, tokenizer_name=args.tokenizer, token_size=args.token_size)

    doc_lengths = [end - start + 1 for (_s, start, end) in chosen]  # +1 approximates appended EOS when missing
    sel = np.asarray(doc_lengths)
    achieved_counts = [int(c) for c in np.histogram(sel, bins=edges)[0]]
    stats = {
        "kind": "document_preserving_docpack",
        "source": "dolmino_long_docs_length_matched",
        "input": args.dolmino_folder,
        "tokenizer": args.tokenizer,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(eos_id),
        "pad_token": args.pad_token,
        "pad_token_id": int(pad_id),
        "pool_size": 64,
        "shuffle_seed": args.shuffle_seed,
        "selection_seed": args.seed,
        "target_real_tokens": args.target_real_tokens,
        "docs_read": len(chosen),
        "dolmino_docs_total": n_docs_total,
        "skipped_pad_collisions": skipped_pad,
        "band25_stats_json": args.band25_stats_json,
        "histogram_bin_edges": [int(x) for x in edges],
        "band25_histogram_counts": [int(c) for c in counts],
        "achieved_histogram_counts": achieved_counts,
        "bin_availability": availability,
        **write_info,
        **summarize(result, doc_lengths),
    }
    (out_folder / "docpack_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    printable = {k: v for k, v in stats.items() if k not in ("per_window_pad_counts", "bin_availability")}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
