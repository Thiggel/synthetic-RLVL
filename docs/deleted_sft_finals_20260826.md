# Deleted SFT finals inventory — 2026-08-26

User scope decision 2026-08-26: the SFT transfer experiments are out of the paper (synthetic experiments only, plus the long-window midtrain if it lands). These 13 full checkpoints were deleted to free vault space for the 5-arm long-window midtrain. All eval results, samples and analyses derived from them are RETAINED under /home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/ and /home/vault/c107fa/c107fa12/synthetic-RLVL/passk_eval/.

## Deleted
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_control_seed3407
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_control_seed3408
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_logic_band15_seed3407
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_logic_band15_seed3408
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_logic_band25_seed3407
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_logic_band25_seed3408
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_nl_exact_band15_seed3407
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_nl_exact_band15_seed3408
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_nl_exact_band25_seed3407
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_reasoning_mixture_20260821/qwen25_7b_mixdepth_nl_exact_band25_seed3408
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_dolci_docpack_rerun_20260814/qwen25_7b_dolmino_control_docpack_2p5b_dolci_100k_lr5em6/
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_dolci_docpack_rerun_20260814/qwen25_7b_dolmino_logic_docpack_2p5b_dolci_100k_lr5em6/
57G	/home/vault/c107fa/c107fa12/synthetic-RLVL/post_sft_dolci_docpack_rerun_20260814/qwen25_7b_dolmino_nl_exact_docpack_2p5b_dolci_100k_lr5em6/

## Regeneration
mixdepth (10): scripts/slurm/jobs/reasoning_mixture_depth_sft_2026-08-21.slurm, array 0-9, ~6h each on 4xA100 (commit e9b27f3); mixtures at $HPCVAULT/synthetic-RLVL/datasets/reasoning_mixture_20260821/ (RETAINED).
docpack (3): scripts/slurm/jobs/qwen25_docpack_rerun_threeway_post_sft_2026-08-14.slurm; base nanotron checkpoints RETAINED at $HPCVAULT/synthetic-RLVL/nanotron_docpack_rerun/ (86G).
