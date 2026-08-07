#!/usr/bin/env python
"""Pack whole documents into fixed-size token windows for the TokenizedBytes loader.

The active Nanotron training path (TokenizedBytesFolderDataset /
DatatroveFolderDataset) reads a flat token stream in disjoint chunks of
``seq_len + 1`` tokens, so documents written as a flat stream are split across
training windows (44-48% of synthetic derivation documents were split in the
original three-way 5B runs). This tool instead packs whole documents into
windows of exactly ``seq_len + 1`` tokens (greedy bounded first-fit), padding
each window's tail with a dedicated padding token. Because every window is
exactly one loader chunk, window boundaries align with training samples and no
document is ever split.

Padding tokens are loss-masked at train time via the collator's
``padding_label_id`` (see nanotron ``NanosetDatasetsArgs.padding_label_id``).
Documents longer than one window are excluded and counted; the pilot corpus is
generated so this count is zero (depth cap chosen from the measured
compact-document length/depth relation).

Output folder layout matches ``tools/preprocess_data.py`` (datatrove):
``00000_docpack.ds`` (little-endian token ids), ``.ds.index`` (uint64
document-end offsets), ``.ds.metadata`` — plus ``docpack_stats.json`` with the
packing accounting used by the audit gates.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass
class PackedWindow:
    tokens: List[int]
    doc_ids: List[int]
    doc_ends: List[int]  # token offset within window just after each document's EOS


@dataclasses.dataclass
class PackResult:
    windows: List[PackedWindow]
    overlength: List[Dict]
    n_docs_packed: int
    window_len: int
    pad_id: int

    @property
    def pad_counts(self) -> List[int]:
        return [self.window_len - len(w.tokens) for w in self.windows]


def pack_documents(
    docs: Iterable[Tuple[int, Sequence[int]]],
    window_len: int,
    pad_id: int,
    pool_size: int = 64,
) -> PackResult:
    """Greedy bounded first-fit packing of whole documents into fixed windows.

    ``docs`` yields ``(doc_id, token_list)``; token lists must already be
    EOS-terminated. A document is placed into the first open window with
    enough remaining capacity; if none fits and the pool is full, the fullest
    open window is closed. Documents longer than ``window_len`` are excluded
    and recorded. Padding is applied by the caller/writer via ``pad_counts``.
    """
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    open_windows: List[PackedWindow] = []
    closed: List[PackedWindow] = []
    overlength: List[Dict] = []
    n_docs_packed = 0

    for doc_id, tokens in docs:
        n = len(tokens)
        if n == 0:
            raise ValueError(f"empty document {doc_id}")
        if pad_id in tokens:
            raise ValueError(f"document {doc_id} contains the padding token id {pad_id}")
        if n > window_len:
            overlength.append({"doc_id": doc_id, "token_len": n})
            continue
        placed = False
        for w in open_windows:
            if window_len - len(w.tokens) >= n:
                w.tokens.extend(tokens)
                w.doc_ids.append(doc_id)
                w.doc_ends.append(len(w.tokens))
                placed = True
                break
        if not placed:
            if len(open_windows) >= pool_size:
                fullest = max(range(len(open_windows)), key=lambda i: len(open_windows[i].tokens))
                closed.append(open_windows.pop(fullest))
            open_windows.append(PackedWindow(tokens=list(tokens), doc_ids=[doc_id], doc_ends=[n]))
        n_docs_packed += 1

    closed.extend(open_windows)
    return PackResult(
        windows=closed,
        overlength=overlength,
        n_docs_packed=n_docs_packed,
        window_len=window_len,
        pad_id=pad_id,
    )


def shuffle_windows(result: PackResult, seed: int) -> PackResult:
    order = np.random.RandomState(seed).permutation(len(result.windows))
    result.windows = [result.windows[i] for i in order]
    return result


def write_nanoset_folder(
    result: PackResult,
    out_folder: Path,
    tokenizer_name: str,
    token_size: int = 4,
    base_name: str = "00000_docpack",
) -> Dict:
    """Write .ds/.ds.index/.ds.metadata mirroring the datatrove layout."""
    out_folder.mkdir(parents=True, exist_ok=True)
    dtype = {2: np.uint16, 4: np.uint32}[token_size]
    ds_path = out_folder / f"{base_name}.ds"
    idx_path = out_folder / f"{base_name}.ds.index"
    meta_path = out_folder / f"{base_name}.ds.metadata"

    total_tokens = 0
    doc_end_offsets: List[int] = []
    sha = hashlib.sha256()
    with ds_path.open("wb") as handle:
        for w in result.windows:
            padded = w.tokens + [result.pad_id] * (result.window_len - len(w.tokens))
            arr = np.asarray(padded, dtype=dtype)
            if arr.size != result.window_len:
                raise AssertionError("window not exactly window_len after padding")
            buf = arr.tobytes()
            handle.write(buf)
            sha.update(buf)
            for end in w.doc_ends:
                doc_end_offsets.append(total_tokens + end)
            total_tokens += result.window_len
    np.asarray(doc_end_offsets, dtype=np.uint64).tofile(idx_path)
    meta_path.write_text(
        f"{tokenizer_name}|{token_size}\n{total_tokens}\n{total_tokens // 1_000_000} MT",
        encoding="utf-8",
    )
    return {
        "ds_file": str(ds_path),
        "total_tokens": total_tokens,
        "n_windows": len(result.windows),
        "sha256": sha.hexdigest(),
    }


def summarize(result: PackResult, doc_lengths: List[int], depths: Optional[List[int]] = None) -> Dict:
    pad_counts = result.pad_counts
    lens = np.asarray(doc_lengths, dtype=np.int64)
    pads = np.asarray(pad_counts, dtype=np.int64)
    seq_len = result.window_len - 1
    # Loss tokens per window under the training collator: labels are
    # window[1:], padding labels are masked, and window[0] is never a pad.
    loss_tokens = int(sum(seq_len - p for p in pad_counts))
    summary = {
        "window_len": result.window_len,
        "seq_len": seq_len,
        "pad_token_id": result.pad_id,
        "n_docs_packed": result.n_docs_packed,
        "n_windows": len(result.windows),
        "overlength_count": len(result.overlength),
        "overlength": result.overlength[:100],
        "real_tokens": int(result.window_len * len(result.windows) - pads.sum()),
        "pad_tokens": int(pads.sum()),
        "packing_efficiency": float(1.0 - pads.sum() / (result.window_len * max(len(result.windows), 1))),
        "loss_tokens": loss_tokens,
        "mean_loss_tokens_per_window": loss_tokens / max(len(result.windows), 1),
        "doc_len_min": int(lens.min()) if lens.size else None,
        "doc_len_max": int(lens.max()) if lens.size else None,
        "doc_len_mean": float(lens.mean()) if lens.size else None,
        "doc_len_percentiles": {
            str(q): float(np.percentile(lens, q)) for q in (1, 10, 25, 50, 75, 90, 99, 100)
        }
        if lens.size
        else {},
        "docs_per_window_mean": float(np.mean([len(w.doc_ids) for w in result.windows])) if result.windows else None,
        "per_window_pad_counts": pad_counts,
    }
    if depths:
        depth_arr = np.asarray(depths, dtype=np.int64)
        summary["depth_min"] = int(depth_arr.min())
        summary["depth_max"] = int(depth_arr.max())
        summary["docs_per_depth"] = {str(d): int((depth_arr == d).sum()) for d in np.unique(depth_arr)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="JSONL with a 'text' field (optionally 'depth')")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq-len", type=int, default=4096, help="training sequence_length; windows are seq_len+1")
    parser.add_argument("--pad-token", default="<|fim_pad|>", help="padding token literal (must never occur in data)")
    parser.add_argument("--eos-token", default=None, help="defaults to the tokenizer's eos_token")
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--target-real-tokens", type=int, default=None, help="stop reading input once this many real tokens (incl. EOS) are packed")
    parser.add_argument("--token-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512, help="tokenization batch size")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token = args.eos_token or tokenizer.eos_token
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    pad_id = tokenizer.convert_tokens_to_ids(args.pad_token)
    if eos_id is None or pad_id is None or pad_id == eos_id:
        raise SystemExit(f"Bad eos/pad token resolution: eos={eos_id} pad={pad_id}")

    window_len = args.seq_len + 1
    texts: List[str] = []
    depths: List[int] = []
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            texts.append(row["text"])
            depths.append(int(row.get("depth", -1)))

    docs: List[Tuple[int, List[int]]] = []
    doc_lengths: List[int] = []
    used_depths: List[int] = []
    total_real = 0
    stop = False
    for start in range(0, len(texts), args.batch_size):
        if stop:
            break
        batch = texts[start : start + args.batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for offset, ids in enumerate(encoded):
            ids = list(ids) + [eos_id]
            doc_id = start + offset
            docs.append((doc_id, ids))
            doc_lengths.append(len(ids))
            used_depths.append(depths[doc_id])
            total_real += len(ids)
            if args.target_real_tokens is not None and total_real >= args.target_real_tokens:
                stop = True
                break

    result = pack_documents(docs, window_len=window_len, pad_id=pad_id, pool_size=args.pool_size)
    result = shuffle_windows(result, seed=args.shuffle_seed)
    out_folder = Path(args.output_folder).expanduser().resolve()
    write_info = write_nanoset_folder(result, out_folder, tokenizer_name=args.tokenizer, token_size=args.token_size)

    stats = {
        "kind": "document_preserving_docpack",
        "input": str(Path(args.input).resolve()),
        "tokenizer": args.tokenizer,
        "eos_token": eos_token,
        "eos_token_id": int(eos_id),
        "pad_token": args.pad_token,
        "pad_token_id": int(pad_id),
        "pool_size": args.pool_size,
        "shuffle_seed": args.shuffle_seed,
        "target_real_tokens": args.target_real_tokens,
        "docs_read": len(docs),
        **write_info,
        **summarize(result, doc_lengths, used_depths if any(d >= 0 for d in used_depths) else None),
    }
    stats_path = out_folder / "docpack_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    printable = {k: v for k, v in stats.items() if k != "per_window_pad_counts"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
