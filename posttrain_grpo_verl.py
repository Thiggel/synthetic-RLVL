from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import hydra
import ray
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf, open_dict
from synthrlvl.datasets import MaterializedSyntheticDataset

from synthrlvl.config import eval_loop_config_from_grpo, task_config_from_cfg
from synthrlvl.grpo_inprocess_eval import install_grpo_eval_patch
from synthrlvl.grpo_inprocess_train import install_grpo_train_patch
from synthrlvl.task import task_sample_from_materialized_row


def _detect_lora_adapter_checkpoint(model_path: str) -> tuple[str | None, str | None]:
    """
    Returns (base_model_name_or_path, adapter_path) if model_path is a PEFT adapter checkpoint.
    Otherwise returns (None, None).
    """
    path = Path(model_path)
    adapter_cfg = path / "adapter_config.json"
    adapter_weights = path / "adapter_model.safetensors"
    if not (adapter_cfg.exists() and adapter_weights.exists()):
        return None, None
    try:
        payload = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base = payload.get("base_model_name_or_path")
        if isinstance(base, str) and base.strip():
            return base.strip(), str(path.resolve())
    except Exception:
        return None, None
    return None, None


def _has_tokenizer_assets(path: str) -> bool:
    """
    Returns True when `path` appears to contain tokenizer files that
    AutoTokenizer can load directly.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    tokenizer_markers = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    return any((p / marker).exists() for marker in tokenizer_markers)


def _merge_lora_adapter_checkpoint(
    *,
    adapter_path: str,
    merged_output_dir: Path,
) -> str:
    """
    Merge a PEFT LoRA adapter into its base model and write a standalone HF model dir.
    Returns the merged model directory path.
    """
    adapter_dir = Path(adapter_path).resolve()
    adapter_cfg = adapter_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        raise ValueError(f"Expected adapter_config.json at {adapter_cfg}")

    payload = json.loads(adapter_cfg.read_text(encoding="utf-8"))
    base_model_name_or_path = payload.get("base_model_name_or_path")
    if not isinstance(base_model_name_or_path, str) or not base_model_name_or_path.strip():
        raise ValueError(f"Adapter checkpoint at {adapter_dir} missing base_model_name_or_path")
    base_model_name_or_path = base_model_name_or_path.strip()

    merged_dir = merged_output_dir.resolve()
    done_marker = merged_dir / ".merge_complete"
    if done_marker.exists():
        return str(merged_dir)

    merged_dir.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=False,
        torch_dtype="auto",
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)

    tokenizer_source = str(adapter_dir) if _has_tokenizer_assets(str(adapter_dir)) else base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(str(merged_dir))

    done_marker.write_text(
        json.dumps(
            {
                "adapter_path": str(adapter_dir),
                "base_model_name_or_path": base_model_name_or_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(merged_dir)


def _build_verl_records_from_materialized_rows(
    rows: list[dict[str, Any]],
    *,
    task_cfg,
    train: bool,
    schema: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        sample = task_sample_from_materialized_row(row, cfg=task_cfg)
        # VERL chat templating expects a messages list.
        # Passing plain strings here can result in empty `messages` and a degenerate
        # prompt like just "Assistant:" after templating.
        prompt_messages = [{"role": "user", "content": sample.prompt}]
        out.append(
            {
                "data_source": "synthetic-rlvl",
                "prompt": prompt_messages,
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": sample.answer,
                    "scorer": "exact_answer",
                },
                "extra_info": {
                    "question": sample.prompt,
                    "split": "train" if train else "val",
                    "index": int(row.get("record_index", i)),
                    "template": task_cfg.template.value,
                    "prefill": task_cfg.prefill.value,
                    "schema": schema,
                    "gold_logic_premises": sample.logic_premises,
                    "gold_logic_conclusion": sample.logic_conclusion,
                    "gold_first_modality_lines": sample.gold_first_modality_lines,
                    "depth": int(sample.depth),
                },
            }
        )
    return out


def _generated_ppo_cfg_path() -> Path:
    import verl

    return Path(verl.__file__).resolve().parent / "trainer" / "config" / "_generated_ppo_trainer.yaml"


def _rollout_mode_for_backend(backend: str) -> str:
    del backend
    return "async"


def _build_verl_cfg(cfg: DictConfig, train_file: Path, val_file: Path) -> DictConfig:
    base = OmegaConf.load(str(_generated_ppo_cfg_path()))
    out_dir = Path(cfg.output_dir).resolve() / cfg.run_name
    # Match SFT exactly: feed only the raw prompt text, without "User:"/"Assistant:"
    # role markers or generation prefixes.
    sft_prompt_only_chat_template = (
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% endfor %}"
    )
    raw_model_path = str(cfg.model.path)
    base_model_path, lora_adapter_path = _detect_lora_adapter_checkpoint(raw_model_path)
    effective_model_path = base_model_path or raw_model_path
    # Only use adapter-local tokenizer files when they actually exist.
    # Some adapter checkpoints are LoRA-only and should inherit base tokenizer.
    if lora_adapter_path is not None and _has_tokenizer_assets(raw_model_path):
        effective_tokenizer_path = raw_model_path
    else:
        effective_tokenizer_path = effective_model_path
    print(
        "[posttrain] model bootstrap:"
        f" raw_model_path={raw_model_path}"
        f" effective_model_path={effective_model_path}"
        f" effective_tokenizer_path={effective_tokenizer_path}"
        f" lora_adapter_path={lora_adapter_path}"
    )

    with open_dict(base):
        base.seed = int(cfg.seed)

        base.actor_rollout_ref.model.path = effective_model_path
        base.actor_rollout_ref.model.tokenizer_path = effective_tokenizer_path
        base.actor_rollout_ref.model.trust_remote_code = False
        base.actor_rollout_ref.model.override_config = {"attn_implementation": "flash_attention_2"}
        base.actor_rollout_ref.model.custom_chat_template = sft_prompt_only_chat_template
        base.actor_rollout_ref.model.lora_rank = int(cfg.model.lora_rank)
        base.actor_rollout_ref.model.lora_alpha = int(cfg.model.lora_alpha)
        base.actor_rollout_ref.model.target_modules = "all-linear"
        if lora_adapter_path is not None:
            base.actor_rollout_ref.model.lora_adapter_path = lora_adapter_path

        base.critic.enable = False

        rollout_backend = str(cfg.grpo.rollout_backend)
        base.actor_rollout_ref.rollout.name = rollout_backend
        # VERL's HF rollout does not provide an async implementation.
        base.actor_rollout_ref.rollout.mode = _rollout_mode_for_backend(rollout_backend)
        base.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        base.actor_rollout_ref.rollout.n = int(cfg.grpo.num_rollouts)
        base.actor_rollout_ref.rollout.temperature = float(cfg.grpo.temperature)
        base.actor_rollout_ref.rollout.prompt_length = int(cfg.grpo.max_prompt_length)
        base.actor_rollout_ref.rollout.response_length = int(cfg.grpo.max_response_length)
        max_model_len_cfg = cfg.grpo.get("max_model_len")
        max_model_len = (
            int(max_model_len_cfg)
            if max_model_len_cfg is not None
            else int(cfg.grpo.max_prompt_length) + int(cfg.grpo.max_response_length)
        )
        base.actor_rollout_ref.rollout.max_model_len = max_model_len
        base.actor_rollout_ref.rollout.max_num_batched_tokens = int(cfg.grpo.max_num_batched_tokens)
        base.actor_rollout_ref.rollout.gpu_memory_utilization = float(cfg.grpo.gpu_memory_utilization)
        if "engine_kwargs" not in base.actor_rollout_ref.rollout or base.actor_rollout_ref.rollout.engine_kwargs is None:
            base.actor_rollout_ref.rollout.engine_kwargs = {}
        if "vllm" not in base.actor_rollout_ref.rollout.engine_kwargs or base.actor_rollout_ref.rollout.engine_kwargs.vllm is None:
            base.actor_rollout_ref.rollout.engine_kwargs.vllm = {}
        # RolloutConfig does not expose a top-level tokenizer field in this VERL version.
        # vLLM rollout consumes `engine_kwargs.vllm`.
        base.actor_rollout_ref.rollout.engine_kwargs.vllm["tokenizer"] = effective_tokenizer_path
        base.actor_rollout_ref.rollout.enable_chunked_prefill = True
        base.actor_rollout_ref.rollout.enable_prefix_caching = True
        base.actor_rollout_ref.rollout.val_kwargs.n = 1
        base.actor_rollout_ref.rollout.val_kwargs.do_sample = False
        base.actor_rollout_ref.rollout.val_kwargs.temperature = 0.0

        base.actor_rollout_ref.actor.use_kl_loss = False
        base.actor_rollout_ref.actor.rollout_n = int(cfg.grpo.num_rollouts)
        base.actor_rollout_ref.actor.ppo_mini_batch_size = int(cfg.grpo.num_prompts)
        base.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = int(cfg.optim.micro_batch_size)
        base.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = int(cfg.grpo.max_num_batched_tokens)
        base.actor_rollout_ref.actor.use_dynamic_bsz = True

        base.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = int(cfg.optim.logprob_micro_batch_size)
        base.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu = int(cfg.grpo.max_num_batched_tokens)
        base.actor_rollout_ref.ref.log_prob_use_dynamic_bsz = True

        base.algorithm.adv_estimator = "grpo"
        base.algorithm.use_kl_in_reward = False
        base.algorithm.kl_ctrl.kl_coef = float(cfg.optim.kl_coef)

        base.data.train_files = str(train_file)
        base.data.val_files = str(val_file)
        base.data.train_max_samples = int(cfg.data.train_samples)
        base.data.val_max_samples = int(cfg.data.val_samples)
        base.data.filter_overlong_prompts = False
        base.data.filter_overlong_prompts_workers = 1
        base.data.max_prompt_length = int(cfg.grpo.max_prompt_length)
        base.data.max_response_length = int(cfg.grpo.max_response_length)
        base.data.train_batch_size = int(cfg.grpo.num_prompts)
        base.data.val_batch_size = int(cfg.grpo.num_prompts)
        base.data.return_raw_chat = False
        base.data.apply_chat_template_kwargs = {"chat_template": sft_prompt_only_chat_template}

        base.reward.custom_reward_function.path = str((Path(__file__).resolve().parent / "synthrlvl" / "verl_reward.py").resolve())
        base.reward.custom_reward_function.name = "compute_score"
        base.reward.custom_reward_function.reward_kwargs = {
            "schema": str(cfg.reward.schema),
        }
        base.reward.num_workers = int(cfg.system.get("reward_num_workers", 2))

        base.trainer.project_name = str(cfg.logging.project)
        base.trainer.experiment_name = str(cfg.run_name)
        base.trainer.logger = ["console", "wandb", "file"]
        n_gpus_per_node = int(cfg.system.get("n_gpus_per_node", 1))
        if n_gpus_per_node < 1:
            raise ValueError(f"system.n_gpus_per_node must be >= 1, got {n_gpus_per_node}")
        base.trainer.nnodes = 1
        base.trainer.n_gpus_per_node = n_gpus_per_node
        base.actor_rollout_ref.rollout.n_gpus_per_node = n_gpus_per_node
        base.trainer.total_training_steps = int(cfg.grpo.train_steps)
        base.trainer.total_epochs = 1
        validation_enabled = bool(cfg.validation.get("enabled", True))
        base.trainer.save_freq = int(cfg.validation.save_every)
        base.trainer.test_freq = int(cfg.validation.eval_every) if validation_enabled else int(cfg.grpo.train_steps) + 10
        base.trainer.val_before_train = bool(cfg.validation.get("before_train", True)) if validation_enabled else False
        base.trainer.default_local_dir = str(out_dir)
        max_actor_ckpts = cfg.validation.get("max_actor_ckpt_to_keep", None)
        if max_actor_ckpts is not None and str(max_actor_ckpts).strip().lower() not in {"", "none", "null"}:
            base.trainer.max_actor_ckpt_to_keep = int(max_actor_ckpts)
            base.trainer.max_critic_ckpt_to_keep = int(max_actor_ckpts)
        base.trainer.resume_mode = str(cfg.resume.get("mode", "disable"))
        resume_from_path = cfg.resume.get("from_path")
        if resume_from_path is not None and str(resume_from_path).strip().lower() not in {"", "none", "null"}:
            base.trainer.resume_from_path = str(resume_from_path)
        base.trainer.log_val_generations = int(cfg.log_generations.num_samples) if validation_enabled else 0
        train_gen_every = int(cfg.log_generations.get("train_every", 0) or 0)
        train_dump_jsonl = bool(cfg.log_generations.get("train_dump_jsonl", False))
        if train_gen_every > 0 or train_dump_jsonl:
            base.trainer.rollout_data_dir = str(out_dir / "train_rollout_data")

        base.ray_kwargs.ray_init.num_cpus = int(cfg.system.ray_cpus)
        # The Ray dashboard is not needed for batch Slurm jobs and its
        # MetricsHead startup has caused otherwise valid GPU allocations to
        # fail before training starts on this cluster.
        base.ray_kwargs.ray_init.include_dashboard = False
        if "runtime_env" not in base.ray_kwargs.ray_init or base.ray_kwargs.ray_init.runtime_env is None:
            base.ray_kwargs.ray_init.runtime_env = {}
        base.ray_kwargs.ray_init.runtime_env.working_dir = str(Path(__file__).resolve().parent)
        # Keep Ray's runtime package small. Large local artifacts/logs can make
        # Ray startup appear alive while no GPU workers are created.
        base.ray_kwargs.ray_init.runtime_env.excludes = [
            ".git/**",
            ".pytest_cache/**",
            "__pycache__/**",
            "**/__pycache__/**",
            "logs/**",
            "wandb/**",
            "wandb_artifacts/**",
            "passk_eval/**",
            "runs/**",
            "tmp/**",
            "datasets/**",
        ]
        if "env_vars" not in base.ray_kwargs.ray_init.runtime_env or base.ray_kwargs.ray_init.runtime_env.env_vars is None:
            base.ray_kwargs.ray_init.runtime_env.env_vars = {}
        for key in (
            "HOME",
            "WORK",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "HF_MODULES_CACHE",
            "HF_HUB_DISABLE_XET",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_GROUP",
            "WANDB_RUN_GROUP",
            "WANDB_DIR",
            "WANDB_CACHE_DIR",
            "WANDB_ARTIFACT_DIR",
            "TVM_FFI_CACHE_DIR",
            "TVM_FFI_DISABLE_TORCH_C_DLPACK",
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        ):
            val = os.environ.get(key)
            if val:
                base.ray_kwargs.ray_init.runtime_env.env_vars[key] = val

    return base


def _install_resource_pool_colocation_patch(max_colocate_count: int) -> None:
    if max_colocate_count < 1:
        raise ValueError(f"system.ray_max_colocate_count must be >= 1, got {max_colocate_count}")
    if max_colocate_count == 3:
        return

    from verl.single_controller.ray import base as ray_base

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            self.resource_pool_dict[resource_pool_name] = ray_base.RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=max_colocate_count,
                name_prefix=resource_pool_name,
            )
        self._check_resource_available()

    ray_base.ResourcePoolManager.create_resource_pool = create_resource_pool


@hydra.main(config_path="conf", config_name="posttrain_grpo", version_base=None)
def main(cfg: DictConfig) -> None:
    mat_ds = MaterializedSyntheticDataset()
    if str(cfg.grpo.rollout_backend) != "vllm":
        raise ValueError("This VERL environment supports `grpo.rollout_backend=vllm` only.")
    model_path = str(cfg.model.path) if cfg.model.get("path") is not None else ""
    if not model_path or model_path.lower() in {"none", "null"}:
        raise ValueError("posttrain requires an SFT checkpoint path via `model.path=...` or env `SFT_CHECKPOINT`.")

    merge_adapter_for_rollout = bool(cfg.model.get("merge_adapter_for_rollout", False))
    if merge_adapter_for_rollout:
        base_model_path_probe, lora_adapter_path_probe = _detect_lora_adapter_checkpoint(model_path.strip())
        if lora_adapter_path_probe is None:
            raise ValueError(
                "model.merge_adapter_for_rollout=true requires `model.path` to point to a LoRA adapter checkpoint."
            )
        merged_dir = Path(cfg.output_dir).resolve() / cfg.run_name / "merged_init_model"
        merged_model_path = _merge_lora_adapter_checkpoint(
            adapter_path=lora_adapter_path_probe,
            merged_output_dir=merged_dir,
        )
        with open_dict(cfg):
            cfg.model.path = merged_model_path
        model_path = merged_model_path
        print(f"[posttrain] Using merged init model at: {merged_model_path}")

    base_model_path, lora_adapter_path = _detect_lora_adapter_checkpoint(model_path.strip())
    disallow_base = model_path.strip() == "allenai/Olmo-3-1025-7B" and lora_adapter_path is None
    if disallow_base and not bool(cfg.get("allow_base_model", False)):
        raise ValueError("Refusing to GRPO-train from base model. Set allow_base_model=true only for dummy pipeline runs.")

    # VERL workers error out if both are set.
    if os.environ.get("ROCR_VISIBLE_DEVICES") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    os.environ.setdefault("WANDB_PROJECT", str(cfg.logging.project))
    if cfg.logging.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg.logging.entity)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["SYNTHRLVL_ENABLE_VERL_TRAIN_PATCH"] = "1"
    os.environ["SYNTHRLVL_GRPO_TRAIN_GEN_EVERY"] = str(int(cfg.log_generations.get("train_every", 0) or 0))
    os.environ["SYNTHRLVL_GRPO_TRAIN_GEN_NUM_SAMPLES"] = str(
        int(cfg.log_generations.get("train_num_samples", 0) or 0)
    )
    os.environ["SYNTHRLVL_GRPO_TRAIN_GEN_MAX_CHARS"] = str(
        int(cfg.log_generations.get("train_max_chars", 2000) or 2000)
    )
    os.environ["SYNTHRLVL_GRPO_TRAIN_DUMP_JSONL"] = (
        "1" if bool(cfg.log_generations.get("train_dump_jsonl", False)) else "0"
    )
    os.environ["SYNTHRLVL_VERL_SANITY_ONLY"] = (
        "1" if bool(cfg.debug.get("verl_sanity_only", False)) else "0"
    )
    sanity_prompt_path = cfg.debug.get("verl_sanity_prompt_path")
    if sanity_prompt_path is not None and str(sanity_prompt_path).strip() not in {"", "null", "None"}:
        os.environ["SYNTHRLVL_VERL_SANITY_PROMPT_PATH"] = str(sanity_prompt_path)
    else:
        os.environ.pop("SYNTHRLVL_VERL_SANITY_PROMPT_PATH", None)
    os.environ["SYNTHRLVL_VERL_SANITY_MAX_NEW_TOKENS"] = str(
        int(cfg.debug.get("verl_sanity_max_new_tokens", 256))
    )

    task_cfg = task_config_from_cfg(cfg)
    eval_cfg = eval_loop_config_from_grpo(cfg)

    out_root = Path(cfg.output_dir).resolve() / cfg.run_name / "synthetic_verl_data"
    out_root.mkdir(parents=True, exist_ok=True)
    train_file = out_root / "train.parquet"
    val_file = out_root / "val.parquet"

    if str(cfg.data.source) != "materialized":
        raise ValueError("GRPO now expects materialized data (`data.source=materialized`).")

    train_subset = str(cfg.data.materialized.train_subset)
    if train_subset == "auto":
        train_subset = mat_ds.train_subset_for_max_step(int(cfg.task.train_max_step))
    val_subset = str(cfg.data.materialized.val_subset)
    if val_subset == "auto":
        val_subset = mat_ds.val_subset_name(int(cfg.task.val_max_step))
    train_rows = mat_ds.load_rows(
        subset=train_subset,
        dataset_id=cfg.data.materialized.dataset_id,
        local_root=cfg.data.materialized.local_root,
        split="train",
        limit=int(cfg.data.train_samples),
    )
    val_rows = mat_ds.load_rows(
        subset=val_subset,
        dataset_id=cfg.data.materialized.dataset_id,
        local_root=cfg.data.materialized.local_root,
        split="train",
        limit=int(cfg.data.val_samples),
    )
    train_records = _build_verl_records_from_materialized_rows(
        train_rows, task_cfg=task_cfg, train=True, schema=str(cfg.reward.schema)
    )
    val_records = _build_verl_records_from_materialized_rows(
        val_rows, task_cfg=task_cfg, train=False, schema=str(cfg.reward.schema)
    )

    Dataset.from_list(train_records).to_parquet(str(train_file))
    Dataset.from_list(val_records).to_parquet(str(val_file))

    verl_cfg = _build_verl_cfg(cfg, train_file, val_file)

    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo

    _install_resource_pool_colocation_patch(int(cfg.system.get("ray_max_colocate_count", 3)))

    run_verl_validation = bool(cfg.validation.get("run_verl_validation", False))
    install_grpo_eval_patch(
        task_cfg=task_cfg,
        eval_cfg=eval_cfg,
        enabled=bool(cfg.eval.enabled),
        run_verl_validation=run_verl_validation,
    )
    install_grpo_train_patch(
        train_gen_every=int(cfg.log_generations.get("train_every", 0) or 0),
        train_gen_num_samples=int(cfg.log_generations.get("train_num_samples", 0) or 0),
        train_gen_max_chars=int(cfg.log_generations.get("train_max_chars", 2000) or 2000),
        dump_jsonl=bool(cfg.log_generations.get("train_dump_jsonl", False)),
    )

    manifest = out_root / "run_manifest.json"
    manifest.write_text(json.dumps(OmegaConf.to_container(verl_cfg, resolve=True), indent=2), encoding="utf-8")

    # Ensure patch installation happens inside the remote TaskRunner process where
    # RayPPOTrainer._validate actually executes.
    from verl.trainer.main_ppo import TaskRunner

    task_cfg_remote = task_cfg
    eval_cfg_remote = eval_cfg
    eval_enabled_remote = bool(cfg.eval.enabled)
    run_verl_validation_remote = run_verl_validation
    train_gen_every_remote = int(cfg.log_generations.get("train_every", 0) or 0)
    train_gen_num_samples_remote = int(cfg.log_generations.get("train_num_samples", 0) or 0)
    train_gen_max_chars_remote = int(cfg.log_generations.get("train_max_chars", 2000) or 2000)
    dump_jsonl_remote = bool(cfg.log_generations.get("train_dump_jsonl", False))
    ray_max_colocate_count_remote = int(cfg.system.get("ray_max_colocate_count", 3))

    class SynthTaskRunner(TaskRunner):
        def run(self, config):
            _install_resource_pool_colocation_patch(ray_max_colocate_count_remote)
            install_grpo_eval_patch(
                task_cfg=task_cfg_remote,
                eval_cfg=eval_cfg_remote,
                enabled=eval_enabled_remote,
                run_verl_validation=run_verl_validation_remote,
            )
            install_grpo_train_patch(
                train_gen_every=train_gen_every_remote,
                train_gen_num_samples=train_gen_num_samples_remote,
                train_gen_max_chars=train_gen_max_chars_remote,
                dump_jsonl=dump_jsonl_remote,
            )
            return super().run(config)

    remote_task_runner = ray.remote(num_cpus=1)(SynthTaskRunner)
    run_ppo(migrate_legacy_reward_impl(verl_cfg), task_runner_class=remote_task_runner)


if __name__ == "__main__":
    main()
