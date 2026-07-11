# Nanotron Mixture-Schedule Audit (2026-07-10)

## Conclusion

The corrected Qwen2.5-7B midtraining pilot implements the intended matched
token mixture. Nanotron blends fixed 4,096-token packed chunks. For each 15%
proof condition, the full 8,192-step schedule contains 157,287 proof chunks
and 891,289 normal-text chunks, corresponding to 644,247,552 proof tokens out
of 4,294,967,296 total tokens (`15.000057%`). Logic and NL therefore receive
the same proof-token exposure to within the identity of the source corpus.

No schedule, capacity, or checkpoint-resume bug was found. The three-step
integration smokes realize one proof chunk out of three (`33.3%`) because they
are intentionally too small to represent 15% finely; this smoke-only
granularity does not apply to production.

## Downstream acceptance gate

The installed lm-eval task registry does not contain `folio`; the original
default would therefore fail all downstream jobs during task resolution. The
validated replacement suite is `gsm8k`, `arc_challenge`, `hellaswag`,
`winogrande`, `piqa`, `logiqa`, `fld_default`,
`fld_logical_formula_default`, `bbh`, `mmlu`, and `mmlu_pro`. The paired FLD
tasks measure transfer to independent natural-language and formula-based
deduction surfaces, while MMLU-Pro adds a harder broad-reasoning control.

Smoke `3834728` runs every task at limit one using Qwen2.5-0.5B in both direct
and native-chat modes. Production direct/instruction evals for control, logic,
and NL all have an additional `afterok:3834728` dependency. The eval wrapper
preflights every task/group name and archives incomplete output directories;
the previous directory-exists guard could incorrectly suppress a retry after
lm-eval wrote only `command.json` and then failed.

## Production Configuration

The production job uses:

- sequence length: `4096`
- tensor parallelism: `4`
- data parallelism: `2`
- microbatch size per replica: `4`
- gradient accumulation: `16`
- optimizer steps: `8192`
- global chunks per optimizer step: `4 * 2 * 16 = 128`
- global tokens per optimizer step: `128 * 4096 = 524,288`
- total scheduled chunks: `8192 * 128 = 1,048,576`
- total scheduled tokens: `1,048,576 * 4096 = 4,294,967,296`

The p15 job configuration lists normal text first and corrected BranchProof-v2
logic or NL second, with normalized weights `[0.85, 0.15]`.

## Exact Realized Counts

Counts were recomputed with Nanotron's compiled
`helpers.build_blending_indices` implementation, the same helper called by
`BlendableDataset`:

| Prefix | Normal chunks | Proof chunks | Proof fraction | Proof tokens |
| ---: | ---: | ---: | ---: | ---: |
| First global batch (128 chunks) | 108 | 20 | 15.625000% | 81,920 |
| Checkpoint 4096 (524,288 chunks) | 445,644 | 78,644 | 15.000153% | 322,125,824 |
| Final step 8192 (1,048,576 chunks) | 891,289 | 157,287 | 15.000057% | 644,247,552 |

The final difference from an exact real-valued 15% allocation is 0.6 of one
4,096-token chunk. Logic and NL use the same weights, seed, and total schedule,
so their realized source counts are identical.

## Corpus Capacity

Pretokenized metadata reports:

| Corpus | Available packed tokens | Maximum production use | Wraparound |
| --- | ---: | ---: | --- |
| Normal continuation | 4,804,719,208 | 4,294,967,296 in control; 3,650,719,744 in p15 | No |
| Corrected logic | 1,200,313,814 | 644,247,552 | No |
| Corrected NL | 1,200,312,018 | 644,247,552 | No |

Thus neither the control nor either p15 arm repeats its packed stream during
the planned run.

## Resume Semantics

`BlendableDataset` builds a deterministic full-run index from weights and
seed. Its dataloader resumes at `consumed_train_samples`, while checkpoint
metadata stores `consumed_tokens_per_dataset_folder`. Consumption accounting
uses the absolute optimizer-step interval in that same full index. The normal
control checkpoint at step 4096 independently records exactly 524,288 chunks
and 2,147,483,648 normal tokens.

Consequently, the pre-submitted recovery jobs continue from the next absolute
chunk and preserve both source proportions and per-source offsets. They do not
restart the blend or replay the first half.

## Scientific Interpretation

This is packed continuation midtraining, not whole-example SFT. A source proof
can cross a 4,096-token boundary, and a packed chunk can contain the end of one
record and the start of another. The experimental intervention is therefore
best described as adding 15% formal-proof or matched NL-proof *tokens* to a
normal continuation stream. The logic/NL comparison remains exposure-matched,
but claims should not imply that every optimizer example contains one intact
proof problem.

## Evidence

- Production wrapper:
  `scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm`
- Nanotron dataset construction:
  `../nanotron/src/nanotron/data/tokenized_bytes.py`
- Exact blend helper and resume accounting:
  `../nanotron/src/nanotron/data/nemo_dataset/blendable_dataset.py` and
  `../nanotron/src/nanotron/data/nemo_dataset/helpers.cpp`
- Verified control metadata:
  `$HPCVAULT/synthetic-RLVL/nanotron_midtrain/qwen25_7b_midtrain_control_p0_4p3b/checkpoints/4096/checkpoint_metadata.json`
- Corrected corpus audit:
  `analysis/branchproof_unique_v2_corpus_audit_2026-07-10.json`
