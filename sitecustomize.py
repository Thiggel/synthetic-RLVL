from __future__ import annotations

import os


def _auto_patch_verl_train_logging() -> None:
    if os.environ.get("SYNTHRLVL_ENABLE_VERL_TRAIN_PATCH", "1") not in {"1", "true", "True"}:
        return
    try:
        from synthrlvl.grpo_inprocess_train import install_grpo_train_patch

        install_grpo_train_patch()
    except Exception:
        # Never block process startup on optional logging patching.
        return


_auto_patch_verl_train_logging()
