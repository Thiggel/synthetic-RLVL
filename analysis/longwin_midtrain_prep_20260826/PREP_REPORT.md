# Long-window depth-25 midtrain — prep report (2026-08-26)

Status: **READY FOR REVIEW — all five arms prepared, all audit gates pass, env smoke passed. Midtrains NOT submitted (awaiting your sbatch).**
Bundle: `analysis/longwin_midtrain_prep_20260826/`.
Storage inventory for everything below: `docs/vault_inventory.md`.

## 1. Design

Five arms, identical in everything except the 10% replacement slice. Token
budget identical to the accepted docpack rerun: **2,385 steps × GBS 128 ×
seq 8192 = 2,500,853,760 tokens** (= 4,770 × 128 × 4,096). Document-preserving
packing into 8,193-token windows, `<|fim_pad|>` (151662) loss-masked tail
padding, padding-compensated proof weight solved per arm by the audit so the
realized synthetic share of *loss tokens* is exactly 10%.

| Arm | CONDITION | 10% slice | audit template |
|---|---|---|---|
| 1 | `control` | none — Dolmino only (weight 1.0); the exact docpack-rerun control recipe (grid row 0) | — |
| 2 | `longdoc` | long Dolmino documents, no deep deduction, length-matched to the band-25 logic histogram | `longdoc` |
| 3 | `logic_band25` | band-25 logic traces, SFT-style rendering | `sft_logic` |
| 4 | `nl_exact_band25` | band-25 nl_exact traces, same latent proofs | `sft_nl_exact` |
| 5 | `condensed_logic_band25` | condensed formal rendering of the same latent proofs | `condensed_logic` |

Deltas vs the accepted 4096 rerun wrapper (all deliberate, all in the header
of the midtrain script):

- SEQ_LEN 4096 → **8192**; TRAIN_STEPS 4770 → **2385** (identical token budget).
- MICRO_BATCH_SIZE 4 → **2**, GRAD_ACCUM 16 → **32**. Global batch stays 128
  sequences and tokens-per-microbatch stays 16,384 (2×8192 = 4×4096), so
  per-GPU activation memory is essentially unchanged from the recipe that ran
  on A100-80. **Not yet measured at 8-GPU scale** — see risk 4.
- Warmup 256 → **128**, LR decay start 256 → **128**, LR_DECAY_STEPS 37891 →
  **18946** (token-equivalent, since each step now covers 2× the tokens).
  LR 1e-5 / MIN_LR 1e-6 unchanged.
- CHECKPOINT_INTERVAL 500 → **250** (same ~262M-token cadence).
- Replacement 5% → **10%**; renderings are the SFT-style prompt+target
  documents (identical to `reasoning_mixture_20260821`), **not** the old
  compact midtrain rendering.
- Run roots + data on `$WORK` (vault is over its soft quota: 1052G/1048.6G).

## 2. Traces (arms 3–5)

Fresh materialized corpus, **72,000 rows, depths 1–25 round-robin**,
hard_fsa_schema, branching 4, distractor 0.5, **seed 20260830** — disjoint
from the 3407-family training seeds, from 20260826 (reserved for the
concurrent graded-deduction eval build) and from 20260806 (pass@k sampling).
Built by job 4109603 in 21 min. Logic and nl_exact are rendered from the same
latent proofs, as always.

- Corpus: `$WORK/synthetic-RLVL/datasets/branchproof_unique_v2_longwin_20260826/train_fixedtarget_up_to_25_72000/`
- Rendered JSONLs + stats: `$WORK/synthetic-RLVL/nanotron_data/longwin_band25_20260826/`
- Builder: `scripts/data/build_longwin_trace_jsonls.py` (fail-closed: any
  document over the 8192 window aborts the build).

### Rendered-document token lengths (actual Qwen2.5-7B tokenizer, n=72,000 each)

| corpus | p50 | p95 | p99 | max | mean | total tokens | frac >4096 | **frac >8192** |
|---|---|---|---|---|---|---|---|---|
| logic | 3,749 | 7,357 | 7,673 | 7,727 | 3,856 | 277,660,923 | 0.442 | **0.0000** |
| nl_exact | 3,819 | 7,417 | 7,744 | 7,851 | 3,907 | 281,300,725 | 0.477 | **0.0000** |
| condensed_logic | 1,450 | 2,886 | 3,007 | 3,014 | 1,500 | 108,027,330 | 0.000 | **0.0000** |

Length-matching between arms 3 and 4 is excellent (p50 3,749 vs 3,819 = 1.9%
apart; max 7,727 vs 7,851), reproducing the mixdepth measurement (3,737/3,796)
on an independent seed — **no length confound between logic and NL**, unlike
the old compact midtrain corpus (2,117 vs 1,212, formal 1.75× longer).

The requirement "fraction of trace docs exceeding 8192 must be ~0%" is met
**exactly**: 0/72,000 in all three renderings, with 341 tokens of headroom on
the worst nl_exact document.

## 3. Packed corpora — token counts, doc lengths, split fraction

All packed with the same `pack_document_preserving_nanoset.py`, `--seq-len
8192`, `--shuffle-seed 42`, identical config across arms.

| arm | docs packed | windows | real tokens | pad tokens | packing eff | doc p50 | doc p99 | doc max | overlength (excluded) | **split docs** |
|---|---|---|---|---|---|---|---|---|---|---|
| logic_band25 | 72,000 | 35,492 | 277,732,923 | 13,053,033 | 0.9551 | 3,750 | 7,674 | 7,728 | **0** | **0** |
| nl_exact_band25 | 72,000 | 35,979 | 281,372,725 | 13,403,222 | 0.9545 | 3,820 | 7,745 | 7,852 | **0** | **0** |
| condensed_logic_band25 | 72,000 | 13,311 | 108,099,330 | 957,693 | 0.9912 | 1,451 | 3,008 | 3,015 | **0** | **0** |
| longdoc_control | (job 4113567) | | ~262M target | | | | | | | **0** |

**On "split documents"**: the document-preserving packer cannot split a
document by construction — whole documents are placed into windows, and any
document longer than one window is *excluded* and counted as `overlength`.
Overlength is 0 in every arm, so the split fraction is 0 **and** nothing was
silently dropped. The decoded-batch audit gate re-verifies this independently
on real loader windows (`split_documents_found` must be 0).

**On the 2.5B matching**: the packed corpora above are the **10% replacement
slice**, not 2.5B each. Every arm trains on exactly 2,500,853,760 tokens; the
slice supplies ~250M of loss tokens and the Dolmino stream supplies the rest.
Epochs consumed over the slice: logic ~0.90, nl_exact ~0.89, condensed ~2.31
(condensed documents are 2.6× shorter, so the same 72k proofs yield 108M
rather than 278M tokens). The accepted 4096 rerun ran at 3.12 synthetic
epochs, so 2.31 is within established practice — but note arm 5 sees each
latent proof ~2.6× more often than arms 3/4. If the review prefers epoch
parity over proof-set parity, regenerate arm 5 with ~185k rows (~25 min).

## 4. Condensed formal rendering (arm 5) — measurement & GO

`scripts/data/condensed_formal_rendering.py`. Renderer-only change; the latent
proof, question and answer are untouched. Condensations:

- fully formal document — the numbered NL theory is not restated (the standard
  document states every rule **twice**: NL in `<question>`, FOL in `<premises>`);
- no trivial constants glossary (`c0 = c0`…), no `<conclusion>` block (it
  restates the last proof line);
- predicate glossary collapsed to one line (`A=lime;B=maple;…`), retained
  because the answer is an NL token that must stay grounded;
- ASCII operator spacing removed: `J(c0)&I(c0)->C(c1)`, `;->E`.
- Atom/predicate names were already minimal (`A..Z`, `c0..cN`) and are kept.

**Verifiability**: 100/100 round-trips pass on both the seed-3407 sample and
the fresh 72k corpus — parse the condensed surface back, require exact
equality with the source `premises_fol`/`proof_fol`/predicates/question/answer,
and require `validate_logic_example(..., citation_free=True)`. (Citation-free
is the correct engine mode: BranchProof-unique-v2 gold traces are
citation-free — `metadata["citation_free_gold"]` — and the cited-strict mode
rejects even the raw untouched rows.)

**Token reduction: 0.389×** (mean 1,500 vs 3,856 on the fresh 72k corpus;
p50 ratio 0.387) — a **61.1% reduction**, reproduced exactly on the
independent seed-3407 sample (0.389).

**Arm-5 gate: GO.** p99 = 3,007 ≤ 3,686 (4096 with 10% margin); observed max
3,014 at depth 25, i.e. 26% margin. Per-depth condensed maxima rise linearly
to 3,010 at depth 25, so the whole band fits 4096 comfortably.
Artifacts: `condensed_rendering_measurement.json`,
`condensed_rendering_measurement_fresh72k.json`.

## 5. Long-doc control source (arm 2)

Source chosen: **(a) long documents already inside the Dolmino mix**. The
2k–8k band is richly populated (11.2M Dolmino documents feed ~67k draws), so
the PG19 / Project Gutenberg fallback was **not needed**.

`scripts/data/build_longdoc_control_docpack.py` reads token ids directly from
the packed Dolmino nanoset at `.ds.index` document boundaries (no
decode/re-tokenize drift), buckets documents into the same 250-token bins as
the band-25 logic histogram, and draws per bin in proportion to that histogram
until 262M real tokens. Documents containing the pad token are skipped and
counted. It **fails closed with an explicit "PG19 fallback needed" message**
if any required bin is exhausted.

Histogram match evidence (target vs achieved, per 250-token bin) is written to
`longdoc_control_docpack_stats.json` (`band25_histogram_counts` vs
`achieved_histogram_counts`, plus `bin_availability`) and tabulated by
`scripts/analysis/summarize_longwin_arms.py`. Because the draw is
proportional-by-construction and the supply is ~170× the demand, the achieved
histogram should track the target to sampling noise; **verify this table
before release** — it is the one number in this arm that could silently
degrade the control.

## 6. Build / audit chain

| job | name | state |
|---|---|---|
| 4109603 | longwin_build_band25 | **COMPLETED** — 72k rows, 3 renderings, 3 packs |
| 4109846 | longwin_rebuild_nanoenv | env built and verified (FAILED state is a post-build artifact; note A) |
| 4110088 | longwin_prereq_rebuild | **Dolmino nanoset COMPLETED and intact**; failed at conversion (note B) |
| 4113565 | longwin_prereq_rebuild (retry) | **COMPLETED** (8 min — nanoset correctly skipped, only the conversion redone) |
| 4113566 / 4113729 | longwin_env_smoke | FAILED — caught two real blockers (notes C, D) |
| 4113730 | longwin_build_longdoc | **COMPLETED** (15m23s) — arm-2 pack (note E) |
| 4113731 | longwin_docpack_audit | **COMPLETED** (1m02s) — **all four arms pass all four gates** |
| 4113750 | longwin_env_smoke (retry) | FAILED — MIG OOM, not a code fault (note F) |
| 4113775 | **longwin_env_smoke** (full GPU) | **COMPLETED / PASSED** — release gate cleared (note F) |

**Note A — 4109846 (env).** The 85-minute flash-attn build succeeded and the
job's own verification printed torch 2.6.0+cu124, flash_attn 2.7.4.post1,
nanotron 0.4 with the LR-resume fix, and the datatrove loader. It then died on
`python -m pip freeze || pip freeze`: uv venvs have no pip module and the PATH
`pip` shim has a broken anaconda shebang, which `set -euo pipefail` promoted to
a job failure. Patched to `uv pip freeze --python ... || true`. **Do not
rebuild the env** (~1.5 h of flash-attn compilation for no gain).

**Note B — 4110088, missing `psutil`.** Step 1, the 5.1B Dolmino nanoset,
**COMPLETED and is intact** (20 GB, 5B packed-token gate passed) after 9 h 13 m
of tokenization. Step 2, `convert_hf_to_nanotron`, then died on
`ModuleNotFoundError: No module named 'psutil'`. Installed (7.2.2) and added to
the rebuild script.

**Note C — 4113566, missing `pip`. The first failure an import audit could not
have caught.** `nanotron/src/nanotron/trainer.py:282` calls
`log_libraries_versions` → `torch.utils.collect_env.main()`, which **shells out
to `pip list`**. The uv-built venv had no pip, so `run_lambda` returned None and
init died with `AttributeError: 'NoneType' object has no attribute 'splitlines'`
— inside `DistributedTrainer.__init__`, *before a single training step*. This
would have killed all five midtrains within seconds of starting, after each had
queued for 1–4 days. Fixed by installing `pip==26.2.1`;
`torch.utils.collect_env.get_pretty_env_info()` now returns 5,094 chars. **pip
is not optional for nanotron** — the accepted env must have had it implicitly,
which is precisely why a from-scratch rebuild lost it.

**Note D — 4113729, wrong `datatrove` distribution. My bug, and the most
dangerous one.** My rebuild script installed `datatrove[io,processing]` from
**PyPI** (0.3.0), but `nanotron/pyproject.toml` pins
`datatrove[io,processing]@git+https://github.com/huggingface/datatrove`. PyPI
0.3.0 installs cleanly and *imports* cleanly, so every import audit passed —
but its `DatatroveFolderDataset.__init__` takes only
`(folder_path, seq_len, filename_pattern, recursive, token_size, max_tokens,
shuffle, seed)`. Both the nanotron training path and
`audit_docpack_training_path.py` call it with `data_folder=`,
`return_positions=` and `positions_from_eos_token_id=`. Nanotron's
compatibility fallback only catches a `TypeError` whose message mentions
`folder_path`, so the real error —
`unexpected keyword argument 'return_positions'` — was **re-raised**. The smoke
died on exactly this. Fixed by installing datatrove from git
(**0.10.0**, commit `a649de79`); the script now carries the git URL.

Two follow-on details worth keeping: the git install pulled **numpy 2.4.6**,
which violates nanotron's `numpy<2` pin, so numpy was reverted to **1.26.4**
and datatrove 0.10.0 verified working against it; and this failure would have
hit the **audit** as well as training, so no arm could have been released.

Verification after the fix (login node, real artifacts):

| check | result |
|---|---|
| `DatatroveFolderDataset` params | now includes `data_folder`, `return_positions`, `positions_from_eos_token_id` |
| Dolmino stream @ seq 8192 | constructs, **623,849** windows, `input_ids` (8193,) + `positions` |
| logic_band25 / nl_exact_band25 / condensed_logic_band25 | **35,492 / 35,979 / 13,311** windows — exactly the packer's counts |
| `torch.utils.collect_env.get_pretty_env_info()` | 5,094 chars (note C path) |

**Note E — 4113567, empty shards, and the answer to "does the training loader
tolerate them?"** `build_longdoc_control_docpack.py` died with
`ValueError: cannot mmap an empty file`: datatrove pre-creates one `.ds` per
worker but only workers that received documents write, so the Dolmino nanoset
is **1 non-empty shard (`00000_unshuffled.ds`, 20 GB) plus 15 zero-length
siblings**. That was a bug in *my* packer script's `np.memmap`, now patched to
drop zero-length shards before mapping.

**The datatrove loader itself tolerates them — confirmed empirically, not
assumed.** `DatatroveFileDataset` never mmaps; it takes `fs.size()`, computes
`_len = 0` for an empty file, and opens lazily only in `__getitem__`.
`DatatroveFolderDataset` routes indices through a `cumsum`/`bisect` over
per-file lengths, so a zero-width interval can never be selected. The
construction above over the real Dolmino folder — 15 of 16 shards empty —
built and indexed correctly. So this is **not** a training risk; it was a
risk only to my own packer.

**Note F — 4113750 MIG OOM, and what the passing smoke does and does not
prove.** After the datatrove fix the smoke got past the dataloader and died in
a 9.75 GiB `a100small` MIG slice with `CUDA out of memory. Tried to allocate
4.64 GiB`. Not a code fault: the smoke model is deliberately tiny (hidden 128,
2 layers) but keeps the **real** 152,064-token vocabulary, so the logits tensor
is `micro_batch x SEQ_LEN x 152064` — about 4.6 GiB at seq 8192. That is a
property of the 8192 window, not of the model, so the smoke was moved to a full
GPU (`a40,a100`, `gres=gpu:1`) and **passed on the first attempt: 4113775,
COMPLETED in 1m12s on an A40, 13,436 MiB peak.**

It genuinely trained rather than exiting early — 3 optimizer steps at
`sequence_length: 8192`, real gradient norms (1.1 / 3.78 / 0.884), `lm_loss`
12 -> 11.9, blending the rebuilt Dolmino stream with the real `logic_band25`
docpack at weights 0.9 / 0.1.

**Scope limit — one claim in an earlier draft of this report was wrong and is
corrected here.** That draft said the smoke exercised "padding-label masking
on". It did not: `nanotron_qwen25_tiny_nanoset_smoke_2026-07-03.slurm` never
emits a `padding_label_id` field (0 occurrences in the file), so despite the
wrapper exporting `NANOSET_PADDING_LABEL_ID=151662` the run logged
`padding_label_id=None`. The smoke therefore validates: the rebuilt env, the
8193-window loader on real packs, dataset blending, and a real optimizer step
at the production sequence length. It does **not** validate padding-loss
masking. That path is covered instead by the audit's `padding_loss_mask` gate,
which drives the actual training collator
(`DataCollatorForCLMWithPositionIds` with `padding_label_id` set) over the real
packed windows and passes on all four arms (section 9). Coverage is complete
across the two checks combined, but neither covers it alone.

Failure history (all resolved, recorded so it is not rediscovered): 4107655/
4107718 died on the deleted venv; 4107741 on `module: command not found` in
batch shells (nvcc is now resolved by Spack path); 4109435 **OUT_OF_MEMORY**
— MAX_JOBS defaulted to SLURM_CPUS_PER_TASK=16 and 16 parallel nvcc compiles
hit MaxRSS 113 GB against a 60000M a40 allocation; 4109442 NODE_FAIL on a0533.
Memory model on alex: RAM is per-CPU and capped (a40 3750 MB/CPU, a100
7500 MB/CPU), cpus-per-gpu is capped, and `--mem` is **rejected outright** on
GPU jobs — the only levers are partition and MAX_JOBS. The env script now
pins `--partition=a100` and `MAX_JOBS=6` (≈45 GB peak against 120 GB).
Every longwin script now carries `--exclude=a0531,a0532,a0533,a0934`.

Audit results: **see §9** (appended when 4113568 lands).

The audit script gained four template modes — `sft_logic`, `sft_nl_exact`,
`condensed_logic`, `longdoc` — with SFT-style structural markers and
`</answer>`-terminal checks (`longdoc` checks structure/padding/mixture only,
since ordinary prose has no markers). Existing `logic`/`nl_exact`/`real_logic`
templates are untouched.

## 7. RELEASE GATES — status

1. **Audit `all_pass=true` on all four packed arms — SATISFIED** (4113731;
   section 9). Each midtrain is additionally fail-closed on its own audit JSON
   and refuses to start if the blend size does not match, so this is enforced
   mechanically at run time, not by discipline.
2. **Env smoke PASSED — SATISFIED** (4113775; note F). This gate paid for
   itself three times over: it is what caught the missing `pip` (note C) and
   the wrong `datatrove` distribution (note D), neither of which any import
   audit could detect, and both of which would have killed all five midtrains
   seconds into a run that had queued for 1-4 days. Note its scope limit in
   note F.
3. **Long-doc histogram match inspected — SATISFIED, with a caveat**
   (section 9). The match is excellent; the *supply headroom* behind it is
   thin. See risk 11.
4. **Still open — first midtrain pass must be watched for OOM in its first
   ~50 steps** (risk 4). The 8192 memory argument is reasoned and now
   partially evidenced (the A40 smoke ran a full 8192 window), but has not
   been demonstrated at 8-GPU TP4/DP2 scale with the real 7.6B model.

## 8. Exact sbatch commands — CLEARED TO SUBMIT

Release gates 1-3 (section 7) are satisfied: all four audits pass and the env
smoke passed. Gate 4 is a during-run check rather than a pre-submit one —
watch the first arm's first ~50 steps for OOM. Sample decoded loader windows
were inspected for every arm before this clearance: condensed documents render
as intended and terminate at `</answer>`, and the long-doc control is ordinary
Dolmino prose with no deductive structure.

Midtrains — `scripts/slurm/jobs/nanotron_qwen25_longwin_midtrain_2026-08-26.slurm`,
singleton-serialized so at most 8 GPUs are used by this line at once:

```bash
cd ~/synthetic-RLVL

sbatch --export=ALL,CONDITION=control                --dependency=singleton --job-name=q25_longwin_midtrain scripts/slurm/jobs/nanotron_qwen25_longwin_midtrain_2026-08-26.slurm
sbatch --export=ALL,CONDITION=longdoc                --dependency=singleton --job-name=q25_longwin_midtrain scripts/slurm/jobs/nanotron_qwen25_longwin_midtrain_2026-08-26.slurm
sbatch --export=ALL,CONDITION=logic_band25           --dependency=singleton --job-name=q25_longwin_midtrain scripts/slurm/jobs/nanotron_qwen25_longwin_midtrain_2026-08-26.slurm
sbatch --export=ALL,CONDITION=nl_exact_band25        --dependency=singleton --job-name=q25_longwin_midtrain scripts/slurm/jobs/nanotron_qwen25_longwin_midtrain_2026-08-26.slurm
sbatch --export=ALL,CONDITION=condensed_logic_band25 --dependency=singleton --job-name=q25_longwin_midtrain scripts/slurm/jobs/nanotron_qwen25_longwin_midtrain_2026-08-26.slurm
```

Use `--export=ALL,CONDITION=…` (not a leading `CONDITION=… sbatch`), so the
value is recorded in the job's environment rather than inherited from the
submitting shell.

Each arm needs **2–3 serialized 24h passes** (the 4096 rerun needed ~2–2.5
days per condition; 8192 steps are ~2× slower but there are half as many).
Submit each CONDITION 2–3× — exactly as the rerun submitted 9 jobs for 3
conditions — or resubmit on timeout. A completed condition exits immediately
with "skipping", so extra submissions are harmless.

Follow-on SFT — `scripts/slurm/jobs/qwen25_longwin_post_sft_2026-08-26.slurm`,
after all five midtrain finals (step 2385) exist:

```bash
sbatch scripts/slurm/jobs/qwen25_longwin_post_sft_2026-08-26.slurm   # array 0-4%2
```

Diff vs the accepted reference `qwen25_docpack_rerun_threeway_post_sft_2026-08-14.slurm`
(the four historical bug classes were checked explicitly):

| aspect | reference | this script |
|---|---|---|
| `--full-parameter` + FSDP + `--gradient-checkpointing` | present | **kept verbatim** |
| `final/config.json` test + `rm -rf checkpoint-*` | present | **kept verbatim** |
| `PYTORCH_CUDA_ALLOC_CONF` | absent | **added** `expandable_segments:True` (mixdepth: 78.9G→69.2G peak) |
| `--save-steps` | hardcoded 250 | **parameterized**, default `SAVE_STEPS=100000` ⇒ only `final/` is written |
| arms / step / verify seq-len | 3 / 4770 / 4096 | 5 / **2385** / **8192** |
| node exclusions | none | `a0531,a0532,a0533,a0934` |

## 9. Audit results — ALL FOUR ARMS PASS ALL FOUR GATES

Job 4113731, seq 8192, 2,385 steps x GBS 128, target ratio 0.10.
Full tables: `ARMS_SUMMARY.md` / `arms_summary.json`; per-arm detail in
`audit_docpack_<arm>.{json,md}`; decoded loader windows in
`decoded_windows_<arm>.md`.

| arm | all_pass | zero_overlen | decoded_batch | padding_mask | exact_mixture | proof_weight | realized ratio | synth epochs |
|---|---|---|---|---|---|---|---|---|
| longdoc_control | **True** | True | True | True | True | 0.10324981 | 0.100001 | 0.951 |
| logic_band25 | **True** | True | True | True | True | 0.10420632 | 0.100001 | 0.896 |
| nl_exact_band25 | **True** | True | True | True | True | 0.10427183 | 0.100002 | 0.885 |
| condensed_logic_band25 | **True** | True | True | True | True | 0.10079632 | 0.100000 | 2.312 |

Realized loss-token ratio lands within 2e-6 of the 0.10 spec on every arm
(tolerance 2e-4). Blend total is 2,500,853,760 tokens for all four, identical
to the control and to the accepted 4096 rerun budget.

### Packed corpora (the 10% slice)

| arm | docs | windows | real tokens | pad tokens | eff | doc p50 | doc p99 | doc max | overlength | **split docs** |
|---|---|---|---|---|---|---|---|---|---|---|
| longdoc_control | 68,396 | 33,138 | 262,000,512 | 9,499,122 | 0.9650 | 3,712 | 7,684 | 7,750 | 0 | **0** |
| logic_band25 | 72,000 | 35,492 | 277,732,923 | 13,053,033 | 0.9551 | 3,750 | 7,674 | 7,728 | 0 | **0** |
| nl_exact_band25 | 72,000 | 35,979 | 281,372,725 | 13,403,222 | 0.9545 | 3,820 | 7,745 | 7,852 | 0 | **0** |
| condensed_logic_band25 | 72,000 | 13,311 | 108,099,330 | 957,693 | 0.9912 | 1,451 | 3,008 | 3,015 | 0 | **0** |

Zero overlength and zero split documents on every arm, confirmed twice: by
construction in the packer, and independently by the decoded-batch gate over
real loader windows (`split_documents_found = 0`).

### Long-doc control length match — the point of arm 2

**The match did not degrade; it is excellent.** Against the band-25 logic
reference: p50 3,712 vs 3,750 (1.0% apart), mean 3,832 vs 3,856 (0.6%), max
7,750 vs 7,728. Per-bin agreement across all 30 populated 250-token bins is
within ~3% relative, including the tail (7000-7250: 0.0396 vs 0.0398;
7250-7500: 0.0387 vs 0.0400; 7500-7750: 0.0392 vs 0.0400). One negligible
artifact: bin 7750-8000 shows achieved 0.0002 against target 0.0000, from the
`+1` EOS allowance in the selector's length bookkeeping — 0.02% of documents.

Source was Dolmino as preferred; **PG19 was not needed** and no bin was
exhausted (`skipped_pad_collisions: 0`, drawn from 11,195,395 Dolmino
documents).

**Caveat that matters for anything after this study — see risk 11.** The
match is excellent but the supply behind it is nearly consumed in the tail:

| bin | available | needed | headroom |
|---|---|---|---|
| 5500-5750 | 5,243 | 2,736 | 1.9x |
| 6250-6500 | 3,933 | 2,729 | 1.4x |
| 6500-6750 | 3,621 | 2,736 | 1.3x |
| 7000-7250 | 3,246 | 2,721 | 1.2x |
| 7250-7500 | 3,113 | 2,736 | 1.1x |
| **7500-7750** | **2,920** | **2,736** | **1.07x** |

Arm 2 consumed **94%** of every Dolmino document in the 7,500-7,750 token
band. The build succeeded with ~7% margin; it did not silently degrade,
because the selector fails closed with an explicit "PG19 fallback needed"
error rather than quietly substituting shorter documents.

## 10. Open risks

1. **Rebuilt env drift — has now fired three times; treat the env as
   unproven until the smoke passes.** Transitive versions float, and each gap
   surfaced later and more expensively than the last: `psutil` (missing, killed
   4110088 nine hours in at checkpoint conversion), `pip` (missing, killed the
   smoke inside trainer init — would have killed every midtrain seconds after a
   1–4 day queue), and `datatrove` (PyPI 0.3.0 instead of the git build
   nanotron pins — imports fine, wrong constructor signature, would have killed
   both training *and* the audit). All three are fixed in the live env and in
   `longwin_rebuild_nanotron_env_2026-08-26.slurm`. The general lesson: an
   **import audit is structurally insufficient** — it cannot catch a subprocess
   call (`pip list`), a keyword-argument mismatch reached only at dataloader
   construction, a lazily-imported package inside a training step, or version
   skew that changes numerics without raising. Only the end-to-end smoke can.
   Recovery procedure if a midtrain dies early on an environment error:
   install the package (or correct the distribution), add it to the rebuild
   script, re-run the smoke, and resume — **never rebuild the env**, which
   costs ~1.5 h of flash-attn compilation. Freeze:
   `logs/nanotron_env_freeze_POST_FIX_20260827.txt` (120 packages, captured
   after all three fixes and after the smoke passed; pins datatrove
   @git+...a649de79, numpy 1.26.4, torch 2.6.0+cu124, flash-attn 2.7.4.post1,
   psutil 7.2.2, pip 26.2.1). The earlier
   `logs/nanotron_env_freeze_4109846.txt` predates the fixes -- do not use it.
2. **Dolmino rebuild identity.** The recipe is deterministic (shuffle seed 42,
   5.1B budget) but the **HF revision is not pinned**. The 5B packed-token
   gate re-applies; compare `stats.json` against the accepted
   5,111,201,524 tokens. A different revision would break comparability with
   the 4096 rerun's control — check this before trusting cross-study contrasts.
3. **Arm-5 epoch asymmetry** (§3): 2.31 epochs vs ~0.90 for arms 3/4, because
   condensation shortens documents. Proof-set is identical; exposure count is
   not. Decide explicitly whether to accept or regenerate at ~185k rows.
4. **8192 memory is reasoned, not measured.** Tokens per microbatch are
   unchanged, but attention workspace grows with sequence length. If pass 1
   OOMs, override `MICRO_BATCH_SIZE=1 GRAD_ACCUM=64` (identical effective
   batch). Watch the first ~50 steps of the first arm.
5. **LR-schedule choice.** Warmup/decay were halved to keep the schedule
   token-equivalent. Step-equivalent constants would double warmup tokens.
   Flag at review if the literal rerun constants are preferred.
6. **Arm 5 differs on the prompt side too**, not only the proof surface: the
   condensed document drops the NL question restatement entirely. That is
   inherent to condensation, but it means arm 5 is not a pure
   surface-of-proof manipulation relative to arms 3/4. State this in any
   readout.
7. **Vault is over soft quota** (1052G/1048.6G, 181k/200k files). Everything
   new went to `$WORK`. The largest deletable block is the 568G mixdepth
   post-SFT tree, whose readouts are already accepted (`docs/vault_inventory.md`).
8. Concurrent agent owns `lm_eval_tasks/synthrlvl_ood/` and
   `build_graded_deduction_eval.py` — untouched here; seed 20260826 avoided.
9. `docs/project_log.md` is concurrently edited, so no log entry was written
   by this prep; the main session should add one.
10. Nothing was committed (per instruction). New/modified files are listed in
    section 11.
11. **Arm 2's length match does not scale — the Dolmino long-document tail is
    now ~exhausted.** The 7,500-7,750 bin was drawn at 94% of available supply
    (1.07x headroom; several neighbouring bins 1.1-1.4x). Consequences: (a) any
    increase in token budget, a second long-doc arm, or a longer window would
    overrun the tail and **must** use the PG19/Gutenberg fallback the brief
    specified — the selector will fail closed rather than degrade silently, so
    this surfaces as an error, not as a bad control; (b) the tail documents are
    near-exhaustively sampled rather than randomly subsampled, so re-running
    with a different selection seed returns almost the same documents — the arm
    is effectively deterministic above ~7k tokens and its tail carries little
    independent variation. Neither undermines the present build; both bound
    what can be reused from it. New/modified files are listed in
    §11.

## 11. Files created or modified by this prep

New:
- `scripts/data/condensed_formal_rendering.py`
- `scripts/data/build_longwin_trace_jsonls.py`
- `scripts/data/build_longdoc_control_docpack.py`
- `scripts/analysis/summarize_longwin_arms.py`
- `scripts/slurm/jobs/longwin_rebuild_nanotron_env_2026-08-26.slurm`
- `scripts/slurm/jobs/longwin_rebuild_prereqs_2026-08-26.slurm`
- `scripts/slurm/jobs/longwin_build_band25_2026-08-26.slurm`
- `scripts/slurm/jobs/longwin_build_longdoc_2026-08-26.slurm`
- `scripts/slurm/jobs/longwin_docpack_audit_2026-08-26.slurm`
- `scripts/slurm/jobs/longwin_env_smoke_2026-08-26.slurm`
- `scripts/slurm/jobs/nanotron_qwen25_longwin_midtrain_2026-08-26.slurm`
- `scripts/slurm/jobs/qwen25_longwin_post_sft_2026-08-26.slurm`
- `docs/vault_inventory.md`
- `analysis/longwin_midtrain_prep_20260826/` (this report + measurements)

Modified:
- `scripts/nanotron/audit_docpack_training_path.py` — four new template modes
  (`sft_logic`, `sft_nl_exact`, `condensed_logic`, `longdoc`); existing
  templates unchanged.
