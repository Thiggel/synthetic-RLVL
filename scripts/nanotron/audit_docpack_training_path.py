#!/usr/bin/env python
"""Audit gates for the document-preserving (docpack) midtraining path.

Runs the four preregistered gates against a packed synthetic nanoset folder
(built by ``scripts/data/pack_document_preserving_nanoset.py``) and the
unchanged Dolmino nanoset, using the same dataset/collator classes as
training (DatatroveFolderDataset + DataCollatorForCLMWithPositionIds):

1. zero-overlength  — no packed document exceeds one window; length report.
2. decoded-batch    — decode real loader windows; every window must contain
                      only whole documents (problem -> derivation -> final
                      answer -> EOS) with padding confined to the tail;
                      representative decoded windows are written out.
3. padding-loss-mask— the training collator (with ``padding_label_id``) must
                      mask exactly the padding labels in synthetic windows and
                      leave Dolmino windows completely unmasked.
4. exact-mixture    — replaying the exact BlendableDataset blending indices
                      for the pilot's sample budget, the realized synthetic
                      share of loss tokens must equal the 5% spec within
                      tolerance. Also solves the proof-dataset weight that
                      compensates for padding overhead.

Run inside the training venv (``$WORK/nanotron``) so the audited classes are
the ones the GPU job will import.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stub_single_process_rank() -> None:
    """The collators only query ranks (no collectives); pin them to rank 0."""
    import nanotron.data.clm_collator as clm_collator

    clm_collator.dist.get_rank = lambda pg=None: 0  # type: ignore[assignment]


class _SingleProcessParallelContext:
    pp_pg = None
    cp_pg = None
    context_parallel_size = 1


def build_folder_dataset(folder: str, seq_len: int, eos_id: int, token_size: int = 4, seed: int = 42):
    from datatrove.utils.dataset import DatatroveFolderDataset

    return DatatroveFolderDataset(
        data_folder=folder,
        seq_len=seq_len,
        filename_pattern=".ds",
        recursive=False,
        token_size=token_size,
        shuffle=True,
        seed=seed,
        return_positions=True,
        positions_from_eos_token_id=eos_id,
    )


def make_collator(seq_len: int, pad_id: int | None):
    from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds

    return DataCollatorForCLMWithPositionIds(
        sequence_length=seq_len,
        input_pp_rank=0,
        output_pp_rank=0,
        parallel_context=_SingleProcessParallelContext(),
        use_doc_masking=False,  # matches get_tb_dataloader(use_doc_masking=Qwen2Config._use_doc_masking=False)
        padding_label_id=pad_id,
    )


def gate_zero_overlength(stats: dict) -> dict:
    return {
        "pass": stats["overlength_count"] == 0,
        "overlength_count": stats["overlength_count"],
        "doc_len_min": stats["doc_len_min"],
        "doc_len_mean": stats["doc_len_mean"],
        "doc_len_max": stats["doc_len_max"],
        "doc_len_percentiles": stats["doc_len_percentiles"],
        "window_len": stats["window_len"],
        "depth_range": [stats.get("depth_min"), stats.get("depth_max")],
    }


def gate_decoded_batch(
    dataset,
    tokenizer,
    stats: dict,
    examples_path: Path,
    n_decode: int,
    n_full_examples: int,
    template: str,
) -> dict:
    eos_id = stats["eos_token_id"]
    pad_id = stats["pad_token_id"]
    n = len(dataset)
    rng = np.random.RandomState(1234)
    idxs = sorted(
        set([0, 1, 2, 3, n // 2, n // 2 + 1, n - 2, n - 1] + list(rng.randint(0, n, size=max(n_decode - 8, 0))))
    )
    # SFT-style long-window templates (2026-08-26): documents are the
    # prompt+target SFT rendering (or the condensed formal rendering), and the
    # long-doc control has no structural markers at all.
    sft_style = {
        "sft_logic": ["<question>\n", "<formal>\n", "<proof>\n", "<answer>\n"],
        "sft_nl_exact": ["<question>\n", "<think>\n", "<proof>\n", "<answer>\n"],
        "condensed_logic": ["<cformal>\n", "<premises>\n", "<proof>\n", "<answer>\n"],
        "longdoc": [],
    }
    if template in sft_style:
        required_markers = list(sft_style[template])
    else:
        required_markers = ["\nSolution:\n", "\nFinal answer: "]
    if template == "logic":
        required_markers += ["Derivation:\n", "Definitions:\n", "Formal premises:\n"]
    elif template == "real_logic":
        # Real corpus (ProofWriter/PARARULE/PrOntoQA) carries the same
        # structure under its own section names: Constants:/Predicates:/
        # Premises: instead of the synthetic Definitions:/Formal premises:.
        required_markers += ["Derivation:\n", "Constants:\n", "Premises:\n"]

    failures = []
    n_docs_total = 0
    examples_md = ["# Decoded-batch audit examples (document-preserving docpack loader)\n"]
    for k, idx in enumerate(idxs):
        item = dataset[int(idx)]
        ids = item["input_ids"].tolist()
        problems = []
        if len(ids) != stats["window_len"]:
            problems.append(f"window length {len(ids)} != {stats['window_len']}")
        # padding must be one contiguous tail block
        try:
            first_pad = ids.index(pad_id)
            if any(t != pad_id for t in ids[first_pad:]):
                problems.append("padding is not confined to the window tail")
            content = ids[:first_pad]
        except ValueError:
            content = ids
        if not content:
            problems.append("window has no content")
        else:
            if content[0] in (pad_id, eos_id):
                problems.append("window does not start with a fresh document")
            if content[-1] != eos_id:
                problems.append("window content does not end at a document boundary (split document!)")
        # split into documents at EOS: every document must be complete
        docs, cur = [], []
        for t in content:
            cur.append(t)
            if t == eos_id:
                docs.append(cur)
                cur = []
        if cur:
            problems.append("trailing partial document without EOS")
        window_doc_summaries = []
        for d_i, doc in enumerate(docs):
            text = tokenizer.decode(doc[:-1])
            missing = [m for m in required_markers if m not in text]
            if missing:
                problems.append(f"doc {d_i} missing markers: {missing}")
            if template in ("sft_logic", "sft_nl_exact", "condensed_logic"):
                if not text.rstrip().endswith("</answer>"):
                    problems.append(f"doc {d_i} does not end with </answer>")
            elif template != "longdoc":
                if not text.rstrip().splitlines()[-1].startswith("Final answer:"):
                    problems.append(f"doc {d_i} does not end with a Final answer line")
            window_doc_summaries.append(
                {"tokens": len(doc), "head": text[:80].replace("\n", " "), "tail": text[-60:].replace("\n", " ")}
            )
        n_docs_total += len(docs)
        if problems:
            failures.append({"window": int(idx), "problems": problems})
        if k < n_full_examples:
            examples_md.append(f"\n## Window {idx} ({len(docs)} documents, {len(ids) - len(content)} pad tokens)\n")
            examples_md.append("```\n" + tokenizer.decode(content) + "\n```\n")
        else:
            examples_md.append(f"\n## Window {idx} summary: {json.dumps(window_doc_summaries)}\n")
    examples_path.write_text("".join(examples_md), encoding="utf-8")
    return {
        "pass": not failures,
        "windows_decoded": len(idxs),
        "documents_seen": n_docs_total,
        "split_documents_found": sum(1 for f in failures for p in f["problems"] if "split" in p),
        "failures": failures,
        "examples_file": str(examples_path),
        "dataset_windows": n,
        "windows_match_stats": n == stats["n_windows"],
    }


def gate_padding_loss_mask(dataset, dolmino_dataset, stats: dict, seq_len: int, n_windows: int) -> dict:
    pad_id = stats["pad_token_id"]
    collator = make_collator(seq_len, pad_id)
    collator_unset = make_collator(seq_len, None)
    n = len(dataset)
    idxs = list(range(min(n_windows // 2, n))) + [n - 1 - i for i in range(min(n_windows - n_windows // 2, n - 1))]
    checked = masked_total = pad_label_total = 0
    failures = []
    for idx in idxs:
        item = dataset[int(idx)]
        batch = [{"input_ids": item["input_ids"].numpy(), "positions": item["positions"].numpy()}]
        out = collator(batch)
        label_ids = np.asarray(out["label_ids"])[0]
        label_mask = np.asarray(out["label_mask"])[0]
        is_pad = label_ids == pad_id
        if not np.array_equal(label_mask, ~is_pad):
            failures.append(int(idx))
        masked_total += int((~label_mask).sum())
        pad_label_total += int(is_pad.sum())
        checked += 1
    # Dolmino windows must be completely unaffected by the padding_label_id.
    dolmino_ok = True
    dol_item = dolmino_dataset[0]
    dol_batch = [{"input_ids": dol_item["input_ids"].numpy(), "positions": dol_item["positions"].numpy()}]
    dol_masked = collator(dol_batch)
    dol_plain = collator_unset(dol_batch)
    if not (np.asarray(dol_masked["label_mask"]).all() and np.asarray(dol_plain["label_mask"]).all()):
        dolmino_ok = False
    if not np.array_equal(np.asarray(dol_masked["label_mask"]), np.asarray(dol_plain["label_mask"])):
        dolmino_ok = False
    return {
        "pass": not failures and dolmino_ok,
        "windows_checked": checked,
        "masked_labels": masked_total,
        "pad_labels": pad_label_total,
        "mask_equals_padding_everywhere": not failures,
        "failing_windows": failures,
        "dolmino_mask_unchanged_all_ones": dolmino_ok,
    }


def blend_counts(weights: list[float], size: int):
    from nanotron.data.nemo_dataset import helpers

    dataset_index = np.zeros(size, dtype=np.int16)
    dataset_sample_index = np.zeros(size, dtype=np.int64)
    dataset_num_samples = np.zeros(len(weights), dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    helpers.build_blending_indices(dataset_index, dataset_sample_index, dataset_num_samples, w, len(weights), size, False)
    return dataset_index, dataset_sample_index, dataset_num_samples


def realized_loss_ratio(proof_weight: float, size: int, pad_counts: list[int], seq_len: int):
    _, _, num_samples = blend_counts([1.0 - proof_weight, proof_weight], size)
    n_dol, n_syn = int(num_samples[0]), int(num_samples[1])
    syn_loss = sum(seq_len - pad_counts[i % len(pad_counts)] for i in range(n_syn))
    dol_loss = n_dol * seq_len
    return syn_loss / (syn_loss + dol_loss), n_syn, n_dol, syn_loss, dol_loss


def gate_exact_mixture(stats: dict, seq_len: int, train_steps: int, global_batch_size: int, target: float, tolerance: float, proof_weight: float | None) -> dict:
    pad_counts = stats["per_window_pad_counts"]
    size = train_steps * global_batch_size
    if proof_weight is None:
        # Solve the proof weight compensating for padding overhead by bisection.
        lo, hi = target, min(4 * target, 0.5)
        for _ in range(50):
            mid = (lo + hi) / 2
            r, *_ = realized_loss_ratio(mid, size, pad_counts, seq_len)
            if r < target:
                lo = mid
            else:
                hi = mid
        proof_weight = (lo + hi) / 2
    ratio, n_syn, n_dol, syn_loss, dol_loss = realized_loss_ratio(proof_weight, size, pad_counts, seq_len)
    sample_ratio = n_syn / (n_syn + n_dol)
    return {
        "pass": abs(ratio - target) <= tolerance,
        "target_loss_token_ratio": target,
        "realized_loss_token_ratio": ratio,
        "abs_error": abs(ratio - target),
        "tolerance": tolerance,
        "proof_weight": proof_weight,
        "normal_weight": 1.0 - proof_weight,
        "blend_size_samples": size,
        "synthetic_samples": n_syn,
        "dolmino_samples": n_dol,
        "synthetic_sample_ratio": sample_ratio,
        "synthetic_loss_tokens": syn_loss,
        "dolmino_loss_tokens": dol_loss,
        "synthetic_epochs_consumed": n_syn / stats["n_windows"],
        "note": "loss tokens = label positions contributing loss (padding labels masked; Dolmino windows contribute all seq_len labels, matching the original runs)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed-folder", required=True)
    parser.add_argument("--dolmino-folder", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--template", default="logic", choices=["logic", "nl_exact", "real_logic", "sft_logic", "sft_nl_exact", "condensed_logic", "longdoc"])
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--target-ratio", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument("--proof-weight", type=float, default=None, help="verify this weight instead of solving")
    parser.add_argument("--decode-windows", type=int, default=32)
    parser.add_argument("--full-examples", type=int, default=6)
    parser.add_argument("--mask-windows", type=int, default=64)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--examples-out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    stub_single_process_rank()
    stats = json.loads((Path(args.packed_folder) / "docpack_stats.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = build_folder_dataset(args.packed_folder, args.seq_len, stats["eos_token_id"])
    dolmino = build_folder_dataset(args.dolmino_folder, args.seq_len, stats["eos_token_id"])

    gates = {
        "zero_overlength": gate_zero_overlength(stats),
        "decoded_batch": gate_decoded_batch(
            dataset, tokenizer, stats, Path(args.examples_out), args.decode_windows, args.full_examples, args.template
        ),
        "padding_loss_mask": gate_padding_loss_mask(dataset, dolmino, stats, args.seq_len, args.mask_windows),
        "exact_mixture": gate_exact_mixture(
            stats, args.seq_len, args.train_steps, args.global_batch_size, args.target_ratio, args.tolerance, args.proof_weight
        ),
    }
    verdict = {
        "all_pass": all(g["pass"] for g in gates.values()),
        "packed_folder": args.packed_folder,
        "dolmino_folder": args.dolmino_folder,
        "template": args.template,
        "seq_len": args.seq_len,
        "train_steps": args.train_steps,
        "global_batch_size": args.global_batch_size,
        "packing_efficiency": stats["packing_efficiency"],
        "gates": gates,
    }
    Path(args.out_json).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    md = ["# Document-preserving loader audit\n\n",
          f"- packed folder: `{args.packed_folder}`\n",
          f"- Dolmino folder (unchanged): `{args.dolmino_folder}`\n",
          f"- verdict: **{'ALL GATES PASS' if verdict['all_pass'] else 'FAILED'}**\n\n"]
    for name, g in gates.items():
        md.append(f"## {name}: {'PASS' if g['pass'] else 'FAIL'}\n\n")
        for k, v in g.items():
            if k in ("pass", "failures", "doc_len_percentiles"):
                continue
            md.append(f"- {k}: {v}\n")
        md.append("\n")
    Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print(json.dumps({"all_pass": verdict["all_pass"], **{k: v["pass"] for k, v in gates.items()},
                      "proof_weight": gates["exact_mixture"]["proof_weight"],
                      "realized_ratio": gates["exact_mixture"]["realized_loss_token_ratio"]}, indent=2))
    if not verdict["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
