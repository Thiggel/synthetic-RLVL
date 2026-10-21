"""Figures + tables for the TL midtraining eval (RLVL-next PLAN W3.2): base_ext vs tl_mid, free vs guided decoding."""
import collections
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/tl/evals")
FIG = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/reports/figures")
OUT = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/tl_mid_eval.md")
RUNS = {"base_ext": E / "base_ext_0.8b", "mid (guide 11:43)": E / "tl_mid_0.8b_lr2em5_b64",
        "mid": E / "tl_mid_0.8b_lr2em5_b64_snapB"}


def load(d, m):
    return [json.loads(x) for x in open(d / m / "generations.jsonl")]


def ok(x):
    return x["valid"] and x["correct"]


gens = {(r, m): load(d, m) for r, d in RUNS.items() for m in ("free", "guided")}
lines = ["# TL midtraining eval (2026-10-09)", "",
         "Held-out generator prompts `datasets/tl_mid_20261009/tl_eval.jsonl`; base_ext was run on the first 200.", "",
         "| model | mode | n | valid | correct | valid∧correct | dead end | mean tokens | first-200 valid∧correct |",
         "|---|---|---|---|---|---|---|---|---|"]
for (r, m), g in gens.items():
    n = len(g)
    f = lambda k: sum(bool(x[k]) for x in g) / n  # noqa: E731
    lines.append(f"| {r} | {m} | {n} | {f('valid'):.3f} | {f('correct'):.3f} | {sum(map(ok, g)) / n:.3f} | "
                 f"{f('dead_end'):.3f} | {sum(x['n_tokens'] for x in g) / n:.0f} | {sum(map(ok, g[:200])) / 200:.3f} |")

# Fig 1: headline bars
fig, ax = plt.subplots(figsize=(7, 3.6))
labels, w = list(RUNS), 0.38
for i, m in enumerate(("free", "guided")):
    vals = [sum(map(ok, gens[(r, m)])) / len(gens[(r, m)]) for r in labels]
    bars = ax.bar([k + (i - 0.5) * w for k in range(len(labels))], vals, w, label=m)
    ax.bar_label(bars, [f"{v:.1%}" for v in vals], fontsize=8)
ax.set_xticks(range(len(labels)), labels)
ax.set_ylabel("valid ∧ correct")
ax.set_ylim(0, 1.1)
ax.legend(loc="upper left")
ax.set_title("TL midtraining: held-out generator prompts")
fig.tight_layout()
fig.savefig(FIG / "tl_mid_headline.png", dpi=150)

# Fig 2: per-family accuracy of the midtrained model (families with >= 10 items, sorted by free accuracy)
fam = collections.defaultdict(lambda: {"free": [], "guided": []})
for m in ("free", "guided"):
    for x in gens[("mid", m)]:
        fam[x["family"]][m].append(ok(x))
hard = sorted((k for k, v in fam.items() if not all(v["free"]) or not all(v["guided"])),
              key=lambda k: sum(fam[k]["free"]) / len(fam[k]["free"]))
rest = {m: [o for k, v in fam.items() if k not in hard for o in v[m]] for m in ("free", "guided")}
groups = [(f"{k}\n(n={len(fam[k]['free'])})", fam[k]) for k in hard] + [(f"{len(fam) - len(hard)} other\nfamilies "
                                                                          f"(n={len(rest['free'])})", rest)]
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, m in enumerate(("free", "guided")):
    vals = [sum(v[m]) / len(v[m]) for _, v in groups]
    bars = ax.bar([k + (i - 0.5) * w for k in range(len(groups))], vals, w, label=m)
    ax.bar_label(bars, [f"{v:.0%}" for v in vals], fontsize=8)
ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=8)
ax.set_ylim(0.6, 1.08)
ax.set_ylabel("valid ∧ correct")
ax.legend(loc="lower right")
ax.set_title("midtrained model: families with any failure")
fig.tight_layout()
fig.savefig(FIG / "tl_mid_families.png", dpi=150)

# Fig 3: failure taxonomy (free) + token lengths
err = collections.Counter()
for x in gens[("mid", "free")]:
    if not ok(x):
        e = re.sub(r"^line \d+: ", "", x["error"]).split(":")[0] or "wrong answer"
        err[f"{e} ({'answer right' if x['correct'] else 'answer wrong'})"] += 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
ks = [k for k, _ in err.most_common()]
a1.barh(ks[::-1], [err[k] for k in ks[::-1]])
a1.set_xlabel("# of 1000 items")
a1.set_title("free decoding: first checker error of failures")
for lab, key in (("base_ext free", ("base_ext", "free")), ("mid free", ("mid", "free")), ("mid guided", ("mid", "guided"))):
    a2.hist([x["n_tokens"] for x in gens[key]], bins=40, range=(0, 2048), histtype="step", label=lab, density=True)
a2.set_yscale("log")
a2.set_xlabel("generated tokens (cap 2048)")
a2.legend()
a2.set_title("proof length")
fig.tight_layout()
fig.savefig(FIG / "tl_mid_failures.png", dpi=150)

lines += ["", "Free-decoding failures (mid): " + ", ".join(f"{k}: {v}" for k, v in err.most_common()), ""]
for m in ("free", "guided"):
    bad = [x for x in gens[("mid", m)] if not ok(x)]
    lines.append(f"mid {m} failures by family: {dict(collections.Counter(x['family'] for x in bad))}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
