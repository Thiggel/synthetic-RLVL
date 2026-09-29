#!/usr/bin/env python
"""How much did each SFT run move its weights? Diagnoses the pure-bf16 DDP bug (2026-09-29).

scripts/train_formal_mixture_sft.py loaded the model in bf16. Without --deepspeed the optimizer then
steps bf16 parameters, and an AdamW update of ~lr (5e-6) is rounded away unless |w| < ~1e-3.
ZeRO-2 (9B, ct, x3) keeps fp32 master weights. For each (init, trained) pair, over every 7th
text-model tensor: the fraction of entries whose bf16 value changed, and ||dW|| / ||W||.
Writes analysis/weight_update_frac.json and reports/figures/weight_update_frac.png.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
HUB = Path("/vol/tmp2/laitenbf/hub")
S = Path("/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925")
BASE = {"0.8b": HUB / "models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "2b": HUB / "models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
        "9b": HUB / "models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404"}
P50 = S / "qwen35_2b_dolci_rlvlgen_p50_lr5em6_seed3407/final"
# label, init, trained, optimizer path, steps
PAIRS = [
    ("0.8B p50 sweep", BASE["0.8b"], S / "qwen35_0.8b_dolci_rlvlgen_p50_lr5em6_seed3407/final", "DDP bf16", 781),
    ("2B p00 sweep", BASE["2b"], S / "qwen35_2b_dolci_rlvlgen_p00_lr5em6_seed3407/final", "DDP bf16", 781),
    ("2B p50 sweep", BASE["2b"], P50, "DDP bf16", 781),
    ("2B p50 + lc", P50, S / "qwen35_2b_p50_cont_lc_lr5em6_seed3407/final", "DDP bf16", 157),
    ("2B p50 + ct", P50, S / "qwen35_2b_p50_cont_ct_lr5em6_seed3407/final", "ZeRO-2 fp32 master", 157),
    ("9B p50 sweep", BASE["9b"], S / "qwen35_9b_dolci_rlvlgen_p50_lr5em6_seed3407/final", "ZeRO-2 fp32 master", 781),
]


def index(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if "visual" not in k and "mtp" not in k:
                    out[k.replace("model.language_model.", "model.")] = (f, k)
    return out


def get(ref):
    with safe_open(ref[0], "pt") as s:
        return s.get_tensor(ref[1]).to(torch.bfloat16).float()


def compare(a, b):
    ia, ib = index(a), index(b)
    ch = tot = 0
    rel = []
    for k in sorted(set(ia) & set(ib))[::7]:
        x, y = get(ia[k]), get(ib[k])
        if x.shape != y.shape:
            continue
        ch += (x != y).sum().item()
        tot += x.numel()
        rel.append(((x - y).norm() / (x.norm() + 1e-12)).item())
    rel.sort()
    return {"changed_frac": ch / tot, "median_rel_delta": rel[len(rel) // 2], "n_tensors": len(rel)}


res = []
for label, a, b, opt, steps in PAIRS:
    if not (Path(b) / "config.json").is_file():
        continue
    r = {"label": label, "init": str(a), "trained": str(b), "optimizer": opt, "steps": steps, **compare(a, b)}
    print(json.dumps({k: r[k] for k in ("label", "optimizer", "changed_frac", "median_rel_delta")}), flush=True)
    res.append(r)
(REPO / "analysis/weight_update_frac.json").write_text(json.dumps(res, indent=2) + "\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
col = ["#E45756" if "bf16" in r["optimizer"] else "#54A24B" for r in res]
names = [f"{r['label']}\n({r['steps']} steps)" for r in res]
for ax, key, title in zip(axes, ("changed_frac", "median_rel_delta"),
                          ("fraction of weights changed (bf16 value)", "median per-tensor ||dW|| / ||W||")):
    ax.bar(range(len(res)), [r[key] for r in res], color=col)
    ax.set_xticks(range(len(res)), names, fontsize=7, rotation=35, ha="right")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=.3)
axes[1].set_yscale("log")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#E45756", label="DDP, bf16 parameters (bug)"),
                    Patch(color="#54A24B", label="ZeRO-2, fp32 master weights")], loc="upper right", fontsize=8)
fig.suptitle("SFT runs trained in pure bf16 barely moved their weights", fontsize=11)
fig.tight_layout(rect=(0, 0, .86, 1))
fig.savefig(REPO / "reports/figures/weight_update_frac.png", dpi=140)
