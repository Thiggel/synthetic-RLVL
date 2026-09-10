#!/usr/bin/env python3
"""Could the ProofWriter gain be contamination or template copying?

Two distinct worries:
  1. Contamination: ProofWriter evaluation text appears in the training data.
     Tested by searching every evaluation item's rare n-grams against the whole
     replaced slice.
  2. In-family similarity: the generator's derivations look enough like
     ProofWriter that the model learned a template rather than deduction.
     Quantified by vocabulary overlap and by how often an evaluation item's
     8-grams appear at all.

Run on alex.
"""
import glob
import json
import random
import re
import sys
from collections import Counter

VAULT = "/home/vault/c107fa/c107fa12/synthetic-RLVL"
WORK = "/home/atuin/c107fa/c107fa12/synthetic-RLVL"

EVAL = VAULT + "/datasets/graded_deduction_eval_20260826"
TRACES = WORK + "/nanotron_data/longwin_band25_20260826"


def words(t):
    return re.findall(r"[a-z]+", t.lower())


def ngrams(toks, n):
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


print("loading ProofWriter evaluation items")
eval_texts = []
for f in sorted(glob.glob(EVAL + "/proofwriter_owa_d*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        eval_texts.append(str(r.get("context", "")) + " " + str(r.get("question", "")))
print("  %d evaluation items" % len(eval_texts))
if not eval_texts:
    sys.exit("no ProofWriter eval items found; check the path")

print("loading the replaced slice (the derivations the models actually trained on)")
train_texts = []
for f in sorted(glob.glob(TRACES + "/*_band25.jsonl")):
    name = f.split("/")[-1]
    n = 0
    for line in open(f):
        r = json.loads(line)
        train_texts.append(str(r.get("text", "")))
        n += 1
        if n >= 20000:      # a 20k sample per rendering is ample for overlap
            break
    print("  %s: %d sampled" % (name, n))

print("\nbuilding 8-gram index over the training slice")
train_8 = set()
for t in train_texts:
    train_8 |= ngrams(words(t), 8)
print("  %d distinct training 8-grams" % len(train_8))

hits = 0
per_item_frac = []
for t in eval_texts:
    g = ngrams(words(t), 8)
    if not g:
        continue
    overlap = len(g & train_8)
    per_item_frac.append(overlap / len(g))
    if overlap:
        hits += 1
print("\nCONTAMINATION")
print("  evaluation items sharing any 8-gram with training: %d of %d (%.2f%%)"
      % (hits, len(eval_texts), 100.0 * hits / len(eval_texts)))
print("  mean fraction of an item's 8-grams seen in training: %.4f"
      % (sum(per_item_frac) / len(per_item_frac)))

print("\nIN-FAMILY SIMILARITY")
ev = Counter()
for t in eval_texts:
    ev.update(words(t))
tr = Counter()
for t in train_texts:
    tr.update(words(t))
ev_top = {w for w, _ in ev.most_common(300)}
tr_top = {w for w, _ in tr.most_common(300)}
print("  top-300 vocabulary overlap: %d of 300 (%.1f%%)"
      % (len(ev_top & tr_top), 100.0 * len(ev_top & tr_top) / 300))
ev_only = [w for w, _ in ev.most_common(60) if w not in tr]
print("  frequent ProofWriter words absent from training entirely: %s" % ev_only[:12])
rng = random.Random(0)
print("  a sampled ProofWriter item:  %r" % eval_texts[rng.randrange(len(eval_texts))][:150])
print("  a sampled training document: %r" % train_texts[rng.randrange(len(train_texts))][:150])
