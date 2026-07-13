#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from nanotron.config import Qwen2Config as NanotronQwen2Config
from nanotron.models import init_on_device_and_dtype
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def _ensure_nanotron_examples_on_path() -> None:
    nanotron_root = Path(os.environ.get("NANOTRON_ROOT", "../nanotron")).resolve()
    if str(nanotron_root) not in sys.path:
        sys.path.insert(0, str(nanotron_root))


_ensure_nanotron_examples_on_path()

from examples.llama.convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model  # noqa: E402


def _qwen2_hf_config(config: NanotronQwen2Config) -> Qwen2Config:
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=False).items()}
    attrs["attention_bias"] = getattr(config, "attention_bias", True)
    return Qwen2Config(**attrs)


def _split_qkv(
    qkv: torch.Tensor,
    *,
    part: str,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q_end = num_attention_heads * head_dim
    k_end = q_end + num_key_value_heads * head_dim
    if part == "q":
        return qkv[:q_end]
    if part == "k":
        return qkv[q_end:k_end]
    if part == "v":
        return qkv[k_end:]
    raise ValueError(f"Unknown qkv part: {part}")


def _split_gate_up(gate_up: torch.Tensor, *, gate: bool) -> torch.Tensor:
    split = gate_up.shape[0] // 2
    return gate_up[:split] if gate else gate_up[split:]


def _normalize_tokenizer_config(save_path: Path) -> None:
    config_path = save_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        if "additional_special_tokens" in config:
            raise RuntimeError(
                "tokenizer config contains both extra_special_tokens and additional_special_tokens"
            )
        config["additional_special_tokens"] = config.pop("extra_special_tokens")
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _normalize_model_config(save_path: Path, *, rope_theta: float) -> None:
    """Keep Transformers 5 checkpoints readable by the Transformers 4 eval env."""
    config_path = save_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rope_parameters = config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise RuntimeError("converted config is missing Transformers 5 rope_parameters")
    serialized_theta = rope_parameters.get("rope_theta")
    if serialized_theta is None or float(serialized_theta) != float(rope_theta):
        raise RuntimeError(
            f"converted rope_parameters.rope_theta={serialized_theta!r}, expected {rope_theta}"
        )
    config["rope_theta"] = float(rope_theta)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def convert_checkpoint(checkpoint_path: Path, save_path: Path, *, tokenizer_name: str) -> None:
    with (checkpoint_path / "model_config.json").open("r", encoding="utf-8") as handle:
        model_config = NanotronQwen2Config(**json.load(handle))

    nanotron_model = load_nanotron_model(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        config_cls=NanotronQwen2Config,
    )
    nanotron_state = nanotron_model.state_dict()
    hf_to_nt = get_weight_mapping(model_config, nt_to_hf=False)

    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
        hf_model = Qwen2ForCausalLM._from_config(_qwen2_hf_config(model_config))

    hf_parameters = dict(hf_model.named_parameters())
    missing_mapping = sorted(set(hf_parameters) - set(hf_to_nt))
    if missing_mapping:
        raise RuntimeError(
            "Nanotron-to-HF mapping leaves parameters uninitialized: "
            + ", ".join(missing_mapping)
        )

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            hf_key = f"{module_name_hf}.{param_name_hf}"
            if hf_key not in hf_to_nt:
                continue
            nt_key = hf_to_nt[hf_key]
            value = nanotron_state[nt_key]
            if "qkv_proj" in nt_key:
                proj_name = module_name_hf.split(".")[-1].split("_", 1)[0]
                value = _split_qkv(
                    value,
                    part=proj_name,
                    num_attention_heads=model_config.num_attention_heads,
                    num_key_value_heads=model_config.num_key_value_heads,
                    head_dim=head_dim,
                )
            elif "gate_up_proj" in nt_key:
                value = _split_gate_up(value, gate="gate" in module_name_hf)
            with torch.no_grad():
                param_hf.copy_(value)

    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(save_path)
    _normalize_tokenizer_config(save_path)
    hf_model.save_pretrained(save_path, safe_serialization=True, max_shard_size="5GB")
    _normalize_model_config(save_path, rope_theta=float(model_config.rope_theta))


def upload_folder(save_path: Path, *, repo_id: str, private: bool, token_env: str) -> None:
    token = os.environ.get(token_env) or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(f"No Hugging Face token found in ${token_env}, $HF_TOKEN, or $HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(save_path),
        path_in_repo="",
        commit_message=f"Upload converted Nanotron checkpoint from {save_path.name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Qwen2 Nanotron checkpoint to Hugging Face format.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--cleanup-local", action="store_true")
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint_path, args.save_path, tokenizer_name=args.tokenizer_name)
    if args.repo_id:
        upload_folder(args.save_path, repo_id=args.repo_id, private=bool(args.private), token_env=str(args.token_env))
    if bool(args.cleanup_local):
        import shutil

        shutil.rmtree(args.save_path, ignore_errors=True)


if __name__ == "__main__":
    main()
