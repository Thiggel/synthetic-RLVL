# Dolmino proof-format and post-SFT diagnosis

Date: 2026-07-31

## Decision

Do not cancel or restart the active control/formal/NL 5B series. Preserve the
existing formal step-5000 result and run the identical NL step-5000 readout
when that checkpoint exists. The current series is still a controlled
logic-versus-matched-NL experiment and already contains likelihood-scored
competence gains that cannot be caused by output parsing.

Add an identical, modern supervised-finetuning readout to all three terminal
checkpoints. Keep direct base-model evaluation as a separate readout. Do not
reuse the previous UltraChat LoRA recipe as the post-SFT result.

Do not immediately spend another 5B tokens per modality. First run a short
format/objective pilot. A full corrected logic/NL rerun is triggered only if
the pilot preserves or improves competence while repairing stopping behavior.
The unchanged control run can be reused if the corrected branches retain the
same base checkpoint, Dolmino stream, schedule, token budget, and seed.

## Execution update: 2026-08-03

Terminal direct evaluation is the unchanged final-checkpoint base-model
readout: the same limit-100 standard tasks, stock/tagged context-QA tasks,
conversion and RoPE checks, decoding caps, extraction, retained generations,
and fail-closed audits are applied to control, formal, and NL at step 9537.
Control/formal `3944016_[0-1]` are running; NL `3944017_2` depends on the
terminal continuation.

The control-only post-SFT LR gate is submitted as a fail-cheap chain. CPU job
`3944069` creates and fingerprints disjoint 100K/2,048 single-turn splits from
the pinned Dolci No-Tools revision. Four-GPU full-parameter FSDP smoke
`3944070_0` runs 32 steps and deletes its model only after an acceptance
audit. Pilot `3944071_[0-1]` then compares `2e-6` and `5e-6` at effective
batch 128, one epoch, linear decay, and 3% warmup. The selected recipe is
determined from held-out SFT loss and response-format diagnostics.
Because this control-only LR choice does not inspect downstream benchmark
scores, it can run before the final contamination filter. An explicit
benchmark-prompt n-gram overlap audit/filter remains mandatory before freezing
the shared treatment SFT subset and applying the chosen LR to all three models.

The corrected mixture experiment is required, but no GPU row is released
until decoded training batches prove that every proof document fits within
the context, document boundaries/positions are preserved, copied-context and
padding loss masks match the declared objective, and modality-token counts
match the requested mixture. The smallest gate remains a 0.5B
`{formal, matched NL} x {compact CLM, proof-focused masked}` pilot. A clean
winner triggers the percentage grid; the old full-document format is not
rerun.

## Confirmed problems

### Corpus construction drift

The July 10 plan specified a compact neutral record:

```text
{problem}

Solution:
{trace}

Final answer: {answer}
```

The implemented `neutral_solution` record is materially larger:

```text
{full problem and premise list}

Solution:
Context:
{declarations and another representation of the premises}

Derivation:
{trace}

Conclusion:
{conclusion}

Final answer: {answer}
```

The NL record repeats the natural-language premises under `Context`; the
formal record adds constants, predicates, and formal premises after the full
natural-language problem. This puts substantial causal-LM loss on copied or
re-encoded context rather than on the derivation and answer.

### Sequence-boundary mismatch

The packed formal corpus contains 144,136 documents averaging 3,816.85 tokens;
the matched NL corpus contains 141,932 documents averaging 3,876.10 tokens.
Exact Nanoset index statistics are:

| Source | p50 | p90 | p95 | maximum | documents above 4,096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| formal | 3,715 | 6,997 | 7,321 | 7,688 | 63,409 / 144,136 (44.0%) |
| matched NL | 3,797 | 7,049 | 7,392 | 7,814 | 66,752 / 141,932 (47.0%) |

Nanotron's active `TokenizedBytesFileDataset` reads the pretokenized stream in
fixed `sequence_length + 1` windows. It does not use the document-index file
to construct document-preserving training instances. The CLM collator assigns
loss to every next token and receives no per-document positions or attention
mask from this loader. Almost half of intervention documents therefore exceed
a training window, and arbitrary stream boundaries can also split shorter
documents. Later derivation/answer spans are frequently trained without the
document opening; early spans frequently contain neither the answer nor EOS.

The configured Qwen EOS ID and preprocessing EOS insertion agree. This is not
an incorrect-EOS-ID bug.

### Readout mismatch

The synthetic stream is base-model causal continuation. It teaches EOS after
`Final answer:` inside a long worked document. Stock LongBench prompts instead
end in bare `Answer:`, use no generation stop string, and expect a short
instruction-following response. The formal step-5000 checkpoint starts a new
QA record after the first answer in 91--97% of stock multi-hop generations.
Tagged prompts with an explicit answer boundary nearly eliminate that failure.

This makes stock no-stop multi-hop F1 an unstable base-model interface metric.
It does not invalidate likelihood-scored MMLU or the inspected reasoning
corrections. Multi-hop reporting should retain stock results as a diagnostic
but use a bounded/tagged protocol, first-answer sensitivity score, cap-hit
rate, EOS/stop rate, and invalid-extraction rate.

## Why matched NL remains necessary

Formal and NL use the same envelope and nearly identical length distribution.
The matched NL step-5000 readout is the direct causal discriminator:

- the same stopping regression implicates the long-document/full-loss path;
- control-like NL stopping implicates formal content rather than the envelope;
- repair under identical SFT indicates response-interface drift rather than
  destroyed reasoning ability.

Changing the NL format now would remove that control.

## Corrected-format pilot

Use the current base checkpoint and Dolmino stream. Train only formal and
matched NL branches for a short gate, with the current control checkpoint as
the unchanged reference.

1. Construct a compact record with one copy of the problem, a
   modality-neutral `Solution:` boundary, only the information needed to
   interpret the derivation, and one `Final answer:` boundary. Remove the
   redundant `Context`/`Conclusion` scaffolding and exact NL premise copy.
2. Enforce a hard token-length audit before preprocessing. Either make every
   record fit in the selected context or use a document-aware instance loader.
   Merely increasing the fixed stream window does not preserve document
   boundaries.
3. Compare the current full-document CLM objective with one proof-focused
   instance objective that masks copied problem/context tokens. Treat masking
   as an objective ablation, not silently as the new definition of
   midtraining.
4. Use about 0.5B total training tokens and evaluate at two intermediate
   checkpoints. Require competence, continuation, cap-hit, and raw-generation
   gates before expanding to 5B.

The smallest informative matrix is `{formal, matched NL} x {compact CLM,
proof-focused masked}` plus the already trained control. If compute permits
only one objective first, prioritize compact document-preserving CLM because
it remains closest to the stated midtraining intervention.

## Post-midtraining SFT readout

Instruction tuning after midtraining is standard model development practice.
OLMo 3 explicitly applies SFT after base-model midtraining, followed by DPO
and RLVR. Its current Dolci instruction mixture contains 2.15M examples.
Tulu 3's published 8B reproduction uses full-parameter SFT, effective batch
128, two epochs, learning rate `5e-6`, linear decay, and 3% warmup.

For this study:

1. Apply exactly the same data order, chat template, optimizer, and schedule
   to control, formal, and matched NL checkpoints.
2. Use Qwen's native chat template, assistant-only loss, explicit assistant
   EOS, and packed instances with correct per-instance masks.
3. Use full-parameter SFT for the evidentiary run. LoRA is acceptable for a
   smoke test but should not be the primary transfer result.
4. Start with a fixed, decontaminated 100K-example subset of
   `allenai/Dolci-Instruct-SFT-No-Tools`, one epoch, effective batch 128,
   maximum length 4,096, linear schedule, 3% warmup, and a control-only
   learning-rate pilot at `2e-6` and `5e-6`. Select the recipe using held-out
   SFT loss plus format diagnostics, not treatment-checkpoint benchmark gains.
5. Save adaptation checkpoints near 10K, 30K, and 100K seen examples. If the
   modality deltas survive and behavior is clean, run one full standard SFT
   epoch as the final readout.
6. Source-filter and n-gram audit the SFT data against every reported
   benchmark. Report direct and post-SFT treatment deltas separately; do not
   compare their absolute macros as if they used the same interface.

The previous generic UltraChat branch is not this experiment. It used a
rank-16 LoRA, effective batch one, 10,000 steps, and no shared reasoning
response contract. It removed literal next-document markers but introduced
long repetition and a BBH extraction floor, so it is a failed alignment
diagnostic rather than evidence that post-SFT erases the midtraining effect.

## Primary references

- [OLMo 3 model flow](https://allenai.org/blog/olmo3): midtraining is followed
  by SFT, DPO, and RLVR; Dolci is the corresponding post-training suite.
- [Dolci Instruct SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT):
  the current 2.15M-example OLMo 3 instruction mixture.
- [Tulu 3 reproduction](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md):
  the published 8B full-SFT hyperparameters and effective batch calculation.
- [Open Instruct](https://github.com/allenai/open-instruct): current open
  post-training implementation and contamination tooling.
