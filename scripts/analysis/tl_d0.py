"""W3.8 Dolci-only control (RLVL-next PLAN): GSM8K (<|im_end|> stop) and Dolci held-out CE of tl_mid_d0 (0% TL) vs base_ext
and the midtrained runs mid5 (25% TL), mid3 / mid4 (55% TL); paired bootstrap of every run against d0 and base_ext."""
import json
import pathlib
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/tl/evals")
FIG = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/reports/figures")
OUT = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/tl_d0.md")
# name -> (GSM8K row file, dolci_loss key, TL share)
RUNS = {"base_ext": ("gsm8k_base_ext_base_ext.jsonl", None, "–"),
        "d0": ("gsm8k_mid_d0_h60_mid_d0_h60.jsonl", "mid_d0_h60", "0%"),
        "mid5": ("gsm8k_mid5_h60_r75_v3q25_mid5_h60_r75_v3q25.jsonl", "mid5_h60_r75_v3q25", "25%"),
        "mid3": ("gsm8k_imend_mid3_mid4_mid3_h60_r45.jsonl", "mid3_h60_r45", "55%"),
        "mid4": ("gsm8k_imend_mid3_mid4_mid4_h60_r45_v3q25.jsonl", "mid4_h60_r45_v3q25", "55%")}


def boot(a, b, n=2000, seed=0):
    """Paired bootstrap 95% CI of mean(a) - mean(b)."""
    rng, d = random.Random(seed), [x - y for x, y in zip(a, b)]
    ms = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), ms[int(0.025 * n)], ms[int(0.975 * n)]


rows = {n: [json.loads(x) for x in open(E / f)] for n, (f, _, _) in RUNS.items()}
corr = {n: [int(r["correct"]) for r in v] for n, v in rows.items()}
ce = {}
for n, (_, k, _) in RUNS.items():
    p = E / f"dolci_loss_{k}.json"
    if k and p.exists():
        ce[n] = json.load(open(p))[k]["dolci"]["ce_per_token"]
acc = {n: sum(c) / len(c) for n, c in corr.items()}
ci = {n: boot(c, [0] * len(c)) for n, c in corr.items()}  # bootstrap CI of the accuracy itself
lines = ["# W3.8 Dolci-only control vs base_ext and the midtrained runs (GSM8K test, 1319, greedy, <|im_end|> stop)", "",
         "| model | TL share | GSM8K | 95% CI | mean tokens | hit cap | Dolci CE/tok | − base_ext | − d0 |", "|---|---|---|---|---|---|---|---|---|"]
for n, (_, _, share) in RUNS.items():
    v = rows[n]
    db = "–" if n == "base_ext" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*boot(corr[n], corr["base_ext"]))
    dd = "–" if n == "d0" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*boot(corr[n], corr["d0"]))
    lines.append(f"| {n} | {share} | {acc[n]:.3f} | [{ci[n][1]:.3f}, {ci[n][2]:.3f}] | {sum(r['n_tokens'] for r in v) / len(v):.0f} | "
                 f"{sum(r['n_tokens'] >= 2048 for r in v) / len(v):.1%} | {ce.get(n, float('nan')):.3f} | {db} | {dd} |")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

cols = {"base_ext": "#999999", "d0": "#a6d854", "mid5": "#e78ac3", "mid3": "#fc8d62", "mid4": "#8da0cb"}
names = list(RUNS)
fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
ax.bar(names, [acc[n] for n in names], color=[cols[n] for n in names], edgecolor="k", lw=0.4,
       yerr=[[acc[n] - ci[n][1] for n in names], [ci[n][2] - acc[n] for n in names]], capsize=4)
for x, n in enumerate(names):
    ax.text(x, ci[n][2] + 0.008, f"{acc[n]:.3f}\nTL {RUNS[n][2]}", ha="center", fontsize=8)
ax.set_ylim(0.45, 0.68)
ax.set_ylabel("GSM8K accuracy")
ax.set_title("GSM8K test (1319, greedy), 95% bootstrap CI")
ax = axs[1]
cn = [n for n in names if n in ce]
ax.bar(cn, [ce[n] for n in cn], color=[cols[n] for n in cn], edgecolor="k", lw=0.4)
for x, n in enumerate(cn):
    ax.text(x, ce[n] + 0.002, f"{ce[n]:.3f}", ha="center", fontsize=8)
ax.set_ylim(0.9, 0.96)
ax.set_title("Dolci held-out CE / token (lower is better)")
fig.suptitle("W3.8 Dolci-only control: how much of the GSM8K drop is the Dolci SFT?")
fig.tight_layout()
fig.savefig(FIG / "tl_d0.png", dpi=150)
