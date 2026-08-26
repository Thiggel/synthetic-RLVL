#!/usr/bin/env python
"""Cross-arm summary for the long-window midtrain prep.

Reads every arm's docpack_stats.json plus its audit JSON and emits the
length-matching evidence table, per-arm token accounting, and the
split-document accounting the review needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ["longdoc_control", "logic_band25", "nl_exact_band25", "condensed_logic_band25"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--jsonl-root", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    pack_root, audit_dir = Path(args.pack_root), Path(args.audit_dir)
    jsonl_root = Path(args.jsonl_root)
    rows = {}
    for arm in ARMS:
        stats_p = pack_root / arm / "docpack_stats.json"
        audit_p = audit_dir / f"audit_docpack_{arm}.json"
        row = {"arm": arm}
        if stats_p.exists():
            s = json.loads(stats_p.read_text())
            pct = s.get("doc_len_percentiles", {})
            row.update(
                {
                    "docs": s["n_docs_packed"],
                    "windows": s["n_windows"],
                    "real_tokens": s["real_tokens"],
                    "pad_tokens": s["pad_tokens"],
                    "packing_efficiency": round(s["packing_efficiency"], 5),
                    "window_len": s["window_len"],
                    "overlength_excluded": s["overlength_count"],
                    "split_docs": 0,  # packer never splits: whole docs only
                    "doc_p50": pct.get("50"),
                    "doc_p99": pct.get("99"),
                    "doc_max": s["doc_len_max"],
                    "loss_tokens": s["loss_tokens"],
                }
            )
        if audit_p.exists():
            a = json.loads(audit_p.read_text())
            g = a["gates"]["exact_mixture"]
            row.update(
                {
                    "audit_all_pass": a["all_pass"],
                    "gate_zero_overlength": a["gates"]["zero_overlength"]["pass"],
                    "gate_decoded_batch": a["gates"]["decoded_batch"]["pass"],
                    "gate_padding_mask": a["gates"]["padding_loss_mask"]["pass"],
                    "gate_exact_mixture": g["pass"],
                    "split_documents_found": a["gates"]["decoded_batch"].get("split_documents_found"),
                    "proof_weight": g["proof_weight"],
                    "realized_ratio": g["realized_loss_token_ratio"],
                    "synthetic_epochs": g["synthetic_epochs_consumed"],
                    "blend_total_tokens": g["blend_size_samples"] * (a["seq_len"]),
                    "synthetic_loss_tokens": g["synthetic_loss_tokens"],
                    "dolmino_loss_tokens": g["dolmino_loss_tokens"],
                }
            )
        rows[arm] = row

    # Pre-pack rendered-document length stats (the length-matching evidence).
    length_rows = {}
    for name in ("logic", "nl_exact", "condensed_logic"):
        p = jsonl_root / f"{name}_band25.stats.json"
        if p.exists():
            s = json.loads(p.read_text())
            length_rows[name] = {k: s[k] for k in ("n", "p50", "p95", "p99", "max", "mean", "total_tokens", "frac_gt_4096", "frac_gt_window")}
    ld = pack_root / "longdoc_control" / "docpack_stats.json"
    if ld.exists():
        s = json.loads(ld.read_text())
        pct = s.get("doc_len_percentiles", {})
        length_rows["longdoc_control"] = {
            "n": s["n_docs_packed"], "p50": pct.get("50"), "p95": pct.get("90"),
            "p99": pct.get("99"), "max": s["doc_len_max"], "mean": s["doc_len_mean"],
            "total_tokens": s["real_tokens"], "frac_gt_4096": None, "frac_gt_window": 0.0,
            "target_histogram": s.get("band25_histogram_counts"),
            "achieved_histogram": s.get("achieved_histogram_counts"),
            "bin_edges": s.get("histogram_bin_edges"),
        }

    payload = {"arms": rows, "document_lengths": length_rows}
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = ["# Long-window arms: packing + length-matching summary\n\n",
          "## Per-arm packed corpus (10% replacement slice)\n\n",
          "| arm | docs | windows | real tokens | pad | eff | doc p50 | doc p99 | doc max | overlength excluded | split docs |\n",
          "|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for arm in ARMS:
        r = rows[arm]
        if "docs" not in r:
            md.append(f"| {arm} | (not built) | | | | | | | | | |\n")
            continue
        md.append(
            f"| {arm} | {r['docs']:,} | {r['windows']:,} | {r['real_tokens']:,} | {r['pad_tokens']:,} | "
            f"{r['packing_efficiency']:.4f} | {r['doc_p50']} | {r['doc_p99']} | {r['doc_max']} | "
            f"{r['overlength_excluded']} | {r.get('split_documents_found', 0)} |\n"
        )
    md.append("\n## Audit gates + realized mixture (2.5B blend, seq 8192)\n\n")
    md.append("| arm | all_pass | zero_overlen | decoded | pad_mask | mixture | proof_weight | realized ratio | epochs | blend tokens |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for arm in ARMS:
        r = rows[arm]
        if "audit_all_pass" not in r:
            md.append(f"| {arm} | (not audited) | | | | | | | | |\n")
            continue
        md.append(
            f"| {arm} | {r['audit_all_pass']} | {r['gate_zero_overlength']} | {r['gate_decoded_batch']} | "
            f"{r['gate_padding_mask']} | {r['gate_exact_mixture']} | {r['proof_weight']:.8f} | "
            f"{r['realized_ratio']:.6f} | {r['synthetic_epochs']:.3f} | {r['blend_total_tokens']:,} |\n"
        )
    md.append("\n## Rendered-document token lengths (Qwen2.5-7B tokenizer)\n\n")
    md.append("| corpus | n | p50 | p95 | p99 | max | mean | total tokens | >4096 | >8192 |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for name, s in length_rows.items():
        f4 = "-" if s["frac_gt_4096"] is None else f"{s['frac_gt_4096']:.3f}"
        md.append(
            f"| {name} | {s['n']:,} | {s['p50']} | {s['p95']} | {s['p99']} | {s['max']} | "
            f"{s['mean']:.0f} | {s['total_tokens']:,} | {f4} | {s['frac_gt_window']:.4f} |\n"
        )
    if "longdoc_control" in length_rows and length_rows["longdoc_control"].get("achieved_histogram"):
        s = length_rows["longdoc_control"]
        md.append("\n### Long-doc control histogram match (250-token bins)\n\n")
        md.append("| bin | band-25 logic (target, normalized) | longdoc achieved (normalized) |\n|---|---|---|\n")
        tgt = s["target_histogram"]; ach = s["achieved_histogram"]; edges = s["bin_edges"]
        st, sa = sum(tgt) or 1, sum(ach) or 1
        for i, (t, a) in enumerate(zip(tgt, ach)):
            if t == 0 and a == 0:
                continue
            md.append(f"| {edges[i]}-{edges[i+1]} | {t/st:.4f} | {a/sa:.4f} |\n")
    md.append("\nNote: the document-preserving packer never splits a document across "
              "windows by construction (docs longer than one window are EXCLUDED and "
              "counted as `overlength`), so the split-document fraction is 0 by design; "
              "the decoded-batch audit gate independently verifies this on real loader windows.\n")
    Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print("".join(md))


if __name__ == "__main__":
    main()
