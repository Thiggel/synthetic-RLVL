"""W3.4 replay sweep (RLVL-next PLAN): TL decoding accuracy vs general ability (Dolci held-out CE, GSM8K) per replay share."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/tl/evals")
FIG = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/reports/figures")
OUT = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/tl_replay_sweep.md")
# name -> (replay share of rows, TL eval dir, key in dolci_loss / gsm8k files)
RUNS = {"r00": (0.0, "tl_mid_0.8b_lr2em5_b64_r00"), "r10": (0.10, "tl_mid_0.8b_lr2em5_b64_r10"),
        "r25": (0.25, "tl_mid_0.8b_lr2em5_b64_snapB"), "r45": (0.45, "tl_mid_0.8b_lr2em5_b64_r45")}


def jload(p):
    return json.load(open(p)) if p.exists() else {}


def gsm1000(n):
    """GSM8K accuracy on the first 1000 test items (common subset: r00 was run with N=1000)."""
    p = E / f"gsm8k_{n}_{n}.jsonl"
    if not p.exists():
        return None
    g = [json.loads(x) for x in open(p)][:1000]
    return sum(x["correct"] for x in g) / len(g) if len(g) == 1000 else None


loss = {**jload(E / "dolci_loss.json")}
gsm = {}
for n in ["base_ext", *RUNS]:
    loss.update(jload(E / f"dolci_loss_{n}.json"))
    gsm.update(jload(E / f"gsm8k_{n}.json"))
rows = []
for n, (share, d) in RUNS.items():
    tl = {m: jload(E / d / m / "summary.json").get("all", {}).get("valid_and_correct") for m in ("free", "guided")}
    rows.append((n, share, tl["free"], tl["guided"], loss.get(n, {}).get("dolci", {}).get("ce_per_token"),
                 loss.get(n, {}).get("tlgen", {}).get("ce_per_token"), gsm1000(n),
                 gsm.get(n, {}).get("mean_tokens"), gsm.get(n, {}).get("acc"),
                 gsm.get(n, {}).get("hit_cap")))
base = ("base_ext", None, None, None, loss["base_ext"]["dolci"]["ce_per_token"],
        loss["base_ext"]["tlgen"]["ce_per_token"], gsm1000("base_ext"),
        gsm.get("base_ext", {}).get("mean_tokens"), gsm.get("base_ext", {}).get("acc"),
        gsm.get("base_ext", {}).get("hit_cap"))
f = lambda v, s="{:.3f}": "–" if v is None else s.format(v)  # noqa: E731
lines = ["# W3.4 replay sweep (0.8B, 120k TL traces fixed, Dolci replay rows added)", "",
         "| model | replay | TL free v∧c | TL guided v∧c | Dolci CE/tok | TL CE/tok | GSM8K (first 1000) | GSM8K mean tokens | GSM8K (all 1319) | GSM8K hit 2048 cap |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in [base, *rows]:
    lines.append(f"| {r[0]} | {f(r[1], '{:.0%}')} | {f(r[2])} | {f(r[3])} | {f(r[4])} | {f(r[5], '{:.4f}')} | "
                 f"{f(r[6])} | {f(r[7], '{:.0f}')} | {f(r[8])} | {f(r[9], '{:.0%}')} |")

fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, (k, lab) in zip(axs, ((2, "TL valid∧correct"), (4, "Dolci held-out CE / token"), (6, "GSM8K accuracy (first 1000)"),
                                     (9, "GSM8K answers hitting 2048-token cap"))):
    pts = [(r[1], r[k]) for r in rows if r[k] is not None]
    if k == 2:
        ax.plot(*zip(*pts), "o-", label="free")
        g = [(r[1], r[3]) for r in rows if r[3] is not None]
        ax.plot(*zip(*g), "s--", label="guided")
        ax.set_ylim(0.9, 1.0)
        ax.legend()
    else:
        ax.plot(*zip(*pts), "o-", label="midtrained")
        if base[k] is not None:
            ax.axhline(base[k], color="gray", ls=":", label="base_ext")
        ax.legend()
    if k == 4:
        ax.set_yscale("log")
    ax.set_xlabel("replay share of training rows")
    ax.set_title(lab)
    ax.set_xticks([r[1] for r in rows], [f"{r[1]:.0%}" for r in rows])
fig.suptitle("Replay sweep: TL skill vs general ability")
fig.tight_layout()
fig.savefig(FIG / "tl_replay_sweep.png", dpi=150)
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
