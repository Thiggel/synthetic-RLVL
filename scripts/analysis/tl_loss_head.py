"""W3.3 loss-head ablation (RLVL-next PLAN): full-vocab CE vs 60-row CE vs guide-legal-set CE on TL rows (0.8B, v1 r25 mix)."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/tl/evals")
FIG = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/reports/figures")
OUT = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/tl_loss_head.md")
# head -> (TL eval / run dir, key in dolci_loss / gsm8k files); full = the v1 model (same mix, same recipe)
RUNS = {"full": ("tl_mid_0.8b_lr2em5_b64_snapB", "r25"), "rows60": ("tl_mid_0.8b_lr2em5_b64_h60", "h60"),
        "legal": ("tl_mid_0.8b_lr2em5_b64_hlegal", "hlegal")}


def jload(p):
    return json.load(open(p)) if p.exists() else {}


loss = jload(E / "dolci_loss.json")
gsm = {}
for _, k in RUNS.values():
    loss.update(jload(E / f"dolci_loss_{k}.json"))
    gsm.update(jload(E / f"gsm8k_{k}.json"))
bench = jload(E / "head_loss_bench.json")
rows = []
for head, (d, k) in RUNS.items():
    tl = {m: jload(E / d / m / "summary.json").get("all", {}) for m in ("free", "guided")}
    rows.append((head, tl["free"].get("valid_and_correct"), tl["free"].get("valid"), tl["free"].get("dead_end"),
                 tl["guided"].get("valid_and_correct"), loss.get(k, {}).get("tlgen", {}).get("ce_per_token"),
                 loss.get(k, {}).get("dolci", {}).get("ce_per_token"), gsm.get(k, {}).get("acc"), gsm.get(k, {}).get("tl_leak"),
                 gsm.get(k, {}).get("hit_cap"), bench.get(head, {}).get("ms_per_1k_tok"), bench.get(head, {}).get("peak_gb")))
f = lambda v, s="{:.3f}": "–" if v is None else s.format(v)  # noqa: E731
lines = ["# W3.3 loss-head ablation (0.8B, v1 r25 mix, lr 2e-5, batch 64, 1 epoch)", "",
         "| TL loss head | TL free v∧c | free valid | free dead end | TL guided v∧c | TL CE/tok (full softmax) | "
         "Dolci CE/tok | GSM8K (1319) | GSM8K TL leak | GSM8K hit 2048 cap | bench ms/1k tok | bench peak GB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r[0]} | {f(r[1])} | {f(r[2])} | {f(r[3])} | {f(r[4])} | {f(r[5], '{:.4f}')} | {f(r[6])} | "
                 f"{f(r[7])} | {f(r[8])} | {f(r[9])} | {f(r[10], '{:.0f}')} | {f(r[11], '{:.1f}')} |")

fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
heads = [r[0] for r in rows]
x = range(len(rows))
ax = axs[0]
ax.bar([i - 0.2 for i in x], [r[1] or 0 for r in rows], 0.4, label="free")
ax.bar([i + 0.2 for i in x], [r[4] or 0 for r in rows], 0.4, label="guided")
ax.set_ylim(0, 1.18)
ax.set_title("TL eval valid∧correct (N=1000)")
ax.legend(ncol=2, loc="upper center")
for ax, k, lab in ((axs[1], 6, "Dolci held-out CE / token"), (axs[2], 7, "GSM8K accuracy (N=1319)")):
    ax.bar(x, [r[k] or 0 for r in rows], color="C2")
    ax.set_title(lab)
    lo = min([r[k] for r in rows if r[k] is not None], default=0)
    ax.set_ylim(lo * 0.9, None)
for ax in axs[:3]:
    ax.set_xticks(list(x), heads)
ax = axs[3]
ax.bar(x, [r[9] or 0 for r in rows], color="C3")
ax.set_xticks(list(x), heads)
ax.set_title("GSM8K: share hitting the 2048-token cap")
fig.suptitle("W3.3: which softmax to train TL rows with")
fig.tight_layout()
fig.savefig(FIG / "tl_loss_head.png", dpi=150)
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
