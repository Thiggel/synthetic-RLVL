#!/usr/bin/env python3
"""Audit paired BranchProof JSONL exports against packed Nanoset documents."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any


QUESTION_RE = re.compile(r"<question>\n(.*?)\n</question>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\n(.*?)\n</answer>", re.DOTALL)
CONSTANT_RE = re.compile(r"\bc(\d+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--nanoset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--sample-indices",
        default="0,1,2,7,31,127,511,2047,8191,32767,65535,131071,196607,262143,300000",
    )
    parser.add_argument("--full-pair-check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if not manifest.get("done"):
        raise AssertionError(f"Incomplete export manifest: {path}")
    return manifest


def load_source_records(path: Path, indices: set[int]) -> dict[int, str]:
    records: dict[int, str] = {}
    with path.open() as handle:
        for index, line in enumerate(handle):
            if index in indices:
                records[index] = json.loads(line)["text"]
            if len(records) == len(indices):
                break
    missing = indices - records.keys()
    if missing:
        raise AssertionError(f"Missing source records {sorted(missing)} in {path}")
    return records


def read_metadata(path: Path) -> tuple[str, int, int]:
    lines = path.read_text().splitlines()
    if len(lines) < 2 or "|" not in lines[0]:
        raise AssertionError(f"Malformed Nanoset metadata: {path}")
    tokenizer_name, token_size = lines[0].rsplit("|", 1)
    return tokenizer_name, int(token_size), int(lines[1])


def audit_nanoset_layout(
    folder: Path,
    manifest: dict[str, Any],
    expected_tokenizer: str,
) -> tuple[list[dict[str, Any]], int, int]:
    shards: list[dict[str, Any]] = []
    packed_tokens = 0
    packed_documents = 0
    for metadata_path in sorted(folder.glob("*.ds.metadata")):
        ds_path = metadata_path.with_suffix("")
        index_path = Path(f"{ds_path}.index")
        tokenizer_name, token_size, metadata_tokens = read_metadata(metadata_path)
        if tokenizer_name != expected_tokenizer:
            raise AssertionError(
                f"Tokenizer mismatch in {metadata_path}: {tokenizer_name} != {expected_tokenizer}"
            )
        if token_size != 4:
            raise AssertionError(f"Unexpected token size in {metadata_path}: {token_size}")
        if ds_path.stat().st_size % token_size:
            raise AssertionError(f"Misaligned token file: {ds_path}")
        file_tokens = ds_path.stat().st_size // token_size
        if file_tokens != metadata_tokens:
            raise AssertionError(
                f"Token-count mismatch for {ds_path}: {file_tokens} != {metadata_tokens}"
            )
        if index_path.stat().st_size % 8:
            raise AssertionError(f"Misaligned document index: {index_path}")
        documents = index_path.stat().st_size // 8
        shards.append(
            {
                "ds_path": ds_path,
                "index_path": index_path,
                "tokens": file_tokens,
                "documents": documents,
                "document_start": packed_documents,
                "document_end": packed_documents + documents,
            }
        )
        packed_tokens += file_tokens
        packed_documents += documents

    if not shards:
        raise AssertionError(f"No Nanoset metadata found under {folder}")
    expected_documents = int(manifest["records"])
    expected_tokens = int(manifest["tokens"]) + expected_documents
    if packed_documents != expected_documents:
        raise AssertionError(
            f"Packed document count mismatch for {folder}: {packed_documents} != {expected_documents}"
        )
    if packed_tokens != expected_tokens:
        raise AssertionError(
            f"Packed token count mismatch for {folder}: {packed_tokens} != {expected_tokens}"
        )
    return shards, packed_tokens, packed_documents


def summarize_document_lengths(shards: list[dict[str, Any]]) -> dict[str, int | float]:
    lengths: list[int] = []
    for shard in shards:
        if not shard["documents"]:
            continue
        ends = array.array("Q")
        ends.frombytes(shard["index_path"].read_bytes())
        if sys.byteorder != "little":
            ends.byteswap()
        previous = 0
        for end in ends:
            lengths.append(end - previous)
            previous = end

    if not lengths:
        raise AssertionError("Cannot summarize an empty Nanoset")
    ordered = sorted(lengths)

    def quantile(q: float) -> int:
        return ordered[round(q * (len(ordered) - 1))]

    summary: dict[str, int | float] = {
        "min": ordered[0],
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }
    for threshold in (4096, 7168, 8192):
        count = sum(length > threshold for length in lengths)
        summary[f"gt_{threshold}_count"] = count
        summary[f"gt_{threshold}_rate"] = count / len(lengths)
    return summary


def read_packed_document(shards: list[dict[str, Any]], document_index: int) -> list[int]:
    for shard in shards:
        if shard["document_start"] <= document_index < shard["document_end"]:
            local_index = document_index - shard["document_start"]
            with shard["index_path"].open("rb") as index_handle:
                if local_index == 0:
                    start = 0
                else:
                    index_handle.seek((local_index - 1) * 8)
                    start = struct.unpack("<Q", index_handle.read(8))[0]
                index_handle.seek(local_index * 8)
                end = struct.unpack("<Q", index_handle.read(8))[0]
            with shard["ds_path"].open("rb") as data_handle:
                data_handle.seek(start * 4)
                token_bytes = data_handle.read((end - start) * 4)
            tokens = array.array("I")
            tokens.frombytes(token_bytes)
            if sys.byteorder != "little":
                tokens.byteswap()
            return tokens.tolist()
    raise IndexError(document_index)


def extract_sections(text: str) -> tuple[str, str]:
    question = QUESTION_RE.search(text)
    answer = ANSWER_RE.search(text)
    if question is None or answer is None:
        raise AssertionError("Missing question or answer wrapper")
    return question.group(1), answer.group(1)


def audit_full_pair(logic_path: Path, nl_path: Path, expected_pairs: int) -> dict[str, int]:
    counts = {
        "paired_records_checked": 0,
        "prompt_mismatches": 0,
        "answer_mismatches": 0,
        "wrapper_mismatches": 0,
        "constant_contiguity_failures": 0,
    }
    with logic_path.open() as logic_handle, nl_path.open() as nl_handle:
        for logic_line, nl_line in zip(logic_handle, nl_handle):
            logic = json.loads(logic_line)["text"]
            nl = json.loads(nl_line)["text"]
            logic_question, logic_answer = extract_sections(logic)
            nl_question, nl_answer = extract_sections(nl)
            counts["prompt_mismatches"] += logic_question != nl_question
            counts["answer_mismatches"] += logic_answer != nl_answer
            counts["wrapper_mismatches"] += not (
                "<formal>" in logic
                and "<think>" not in logic
                and "<think>" in nl
                and "<formal>" not in nl
            )
            constants = sorted({int(value) for value in CONSTANT_RE.findall(logic_question)})
            depth = max(constants) if constants else -1
            counts["constant_contiguity_failures"] += constants != list(range(depth + 1))
            counts["paired_records_checked"] += 1
    if counts["paired_records_checked"] != expected_pairs:
        raise AssertionError(
            f"Paired prefix length mismatch: {counts['paired_records_checked']} != {expected_pairs}"
        )
    failures = sum(value for key, value in counts.items() if key != "paired_records_checked")
    if failures:
        raise AssertionError(f"Full paired-prefix audit found {failures} failures: {counts}")
    return counts


def main() -> None:
    args = parse_args()
    sample_indices = {int(value) for value in args.sample_indices.split(",") if value}
    manifests = {
        template: load_manifest(args.data_root / f"{template}.jsonl.manifest.json")
        for template in ("logic", "nl_exact")
    }
    source_paths = {
        template: args.data_root / f"{template}.jsonl" for template in manifests
    }
    sources = {
        template: load_source_records(path, sample_indices)
        for template, path in source_paths.items()
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise AssertionError(f"Tokenizer {args.tokenizer} has no EOS token")

    layouts: dict[str, list[dict[str, Any]]] = {}
    layout_summaries: dict[str, dict[str, Any]] = {}
    for template, manifest in manifests.items():
        shards, packed_tokens, packed_documents = audit_nanoset_layout(
            args.nanoset_root / template, manifest, args.tokenizer
        )
        layouts[template] = shards
        layout_summaries[template] = {
            "source_records": int(manifest["records"]),
            "source_tokens": int(manifest["tokens"]),
            "packed_documents": packed_documents,
            "packed_tokens": packed_tokens,
            "nonempty_shards": sum(shard["tokens"] > 0 for shard in shards),
            "total_shards": len(shards),
            "document_length_tokens": summarize_document_lengths(shards),
        }

    sampled: list[dict[str, Any]] = []
    for index in sorted(sample_indices):
        logic_text = sources["logic"][index]
        nl_text = sources["nl_exact"][index]
        logic_question, logic_answer = extract_sections(logic_text)
        nl_question, nl_answer = extract_sections(nl_text)
        if logic_question != nl_question or logic_answer != nl_answer:
            raise AssertionError(f"Paired source mismatch at record {index}")
        constants = sorted({int(value) for value in CONSTANT_RE.findall(logic_question)})
        depth = max(constants) if constants else -1
        if constants != list(range(depth + 1)):
            raise AssertionError(f"Non-contiguous constants at record {index}: {constants}")

        sample: dict[str, Any] = {"index": index, "depth": depth, "answer": logic_answer}
        for template, text in (("logic", logic_text), ("nl_exact", nl_text)):
            actual_tokens = read_packed_document(layouts[template], index)
            expected_tokens = tokenizer.encode(text, add_special_tokens=False) + [eos_id]
            if actual_tokens != expected_tokens:
                raise AssertionError(
                    f"Packed/source token mismatch for {template} record {index}: "
                    f"{len(actual_tokens)} != {len(expected_tokens)}"
                )
            decoded = tokenizer.decode(actual_tokens[:-1], skip_special_tokens=False)
            if decoded != text:
                raise AssertionError(f"Decode round-trip mismatch for {template} record {index}")
            digest = hashlib.sha256(array.array("I", actual_tokens).tobytes()).hexdigest()
            sample[template] = {
                "tokens_including_eos": len(actual_tokens),
                "sha256": digest,
                "decoded_prefix": decoded[:240],
                "decoded_suffix": decoded[-320:],
            }
        sampled.append(sample)

    pair_audit = None
    if args.full_pair_check:
        pair_audit = audit_full_pair(
            source_paths["logic"],
            source_paths["nl_exact"],
            min(int(manifests["logic"]["records"]), int(manifests["nl_exact"]["records"])),
        )

    result = {
        "accepted": True,
        "tokenizer": args.tokenizer,
        "eos_token_id": eos_id,
        "layouts": layout_summaries,
        "full_pair_audit": pair_audit,
        "sampled_records": sampled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
