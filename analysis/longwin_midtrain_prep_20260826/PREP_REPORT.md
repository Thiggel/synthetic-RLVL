# Long-window depth-25 midtrain — prep report (2026-08-26)

Status: **prepared; corpora built and packed; audits + env smoke running; midtrains NOT submitted.**
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
| longdoc_control | (job 4109857) | | ~262M target | | | | | | | **0** |

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
| 4109846 | longwin_rebuild_nanoenv | running (a100, MAX_JOBS=6) |
| 4109603 | longwin_build_band25 | **COMPLETED** (21m07s) — 72k rows, 3 renderings, 3 packs |
| 4109856 | longwin_prereq_rebuild | pending (afterok 4109846) — Dolmino nanoset + tp1 ckpt |
| 4109857 | longwin_build_longdoc | pending (afterok 4109856) — arm-2 pack |
| 4109858 | longwin_docpack_audit | pending (afterok 4109857) — four gates × four arms |
| 4109859 | **longwin_env_smoke** | pending (afterok 4109856) — **hard release gate** |

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

Audit results: **see §9** (appended when 4109858 lands).

The audit script gained four template modes — `sft_logic`, `sft_nl_exact`,
`condensed_logic`, `longdoc` — with SFT-style structural markers and
`</answer>`-terminal checks (`longdoc` checks structure/padding/mixture only,
since ordinary prose has no markers). Existing `logic`/`nl_exact`/`real_logic`
templates are untouched.

## 7. RELEASE GATES — all must pass before any midtrain is submitted

1. **4109858 audit: `all_pass=true` on all four packed arms** at blend size
   2385 (each midtrain is fail-closed on this JSON and refuses to start
   otherwise, so this is enforced mechanically, not by discipline).
2. **4109859 env smoke: PASSED.** The `$WORK/nanotron` venv was rebuilt from
   scratch and its transitive dependency versions are **not** identical to the
   accepted env. `longwin_env_smoke_2026-08-26.slurm` wraps the accepted
   tiny-nanoset smoke but points it at the **real** artifacts and geometry —
   rebuilt Dolmino as the normal stream, the real `logic_band25` docpack as
   the proof stream, padding-label masking on, **SEQ_LEN 8192** — so it
   exercises the 8193-window loader, the padding mask, and a real optimizer
   step on the rebuilt env. This is a **hard gate**: do not sbatch any midtrain
   until it reports PASSED.
3. **Long-doc histogram table inspected** (§5).
4. First midtrain pass watched for OOM in its first ~50 steps (risk 4).

## 8. Exact sbatch commands (DO NOT run until §7 gates pass)

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

## 9. Audit results

PENDING (job 4109858). Will contain, per arm: `all_pass`, the four gate
verdicts, `proof_weight`, realized loss-token ratio vs the 0.10 target,
synthetic epochs, packing efficiency, `split_documents_found`, and the
long-doc target-vs-achieved histogram. Generate the tables with:

```bash
python scripts/analysis/summarize_longwin_arms.py \
  --pack-root $WORK/synthetic-RLVL/nanosets_longwin_20260826 \
  --audit-dir analysis/longwin_midtrain_prep_20260826 \
  --jsonl-root $WORK/synthetic-RLVL/nanotron_data/longwin_band25_20260826 \
  --out-md analysis/longwin_midtrain_prep_20260826/ARMS_SUMMARY.md \
  --out-json analysis/longwin_midtrain_prep_20260826/arms_summary.json
```

## 10. Open risks

1. **Rebuilt env drift (gated).** Transitive versions float; smoke 4109859 is
   the gate. Freeze recorded in `logs/nanotron_env_freeze_4109846.txt`.
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
