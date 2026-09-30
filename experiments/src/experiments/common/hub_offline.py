"""Hugging Face offline mode for the model runs and the pinned weight files they read from the
local cache."""

import os
from pathlib import Path
from typing import Any

import huggingface_hub
import huggingface_hub.constants
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers.utils import hub as transformers_hub

Record = dict[str, Any]

WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def offline_state() -> Record:
    return {
        "env_HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "huggingface_hub_is_offline_mode": bool(huggingface_hub.is_offline_mode()),
        "transformers_is_offline_mode": bool(transformers_hub.is_offline_mode()),
        "huggingface_hub": huggingface_hub.__version__,
        "transformers": transformers.__version__,
    }


def enforce_offline() -> Record:
    os.environ["HF_HUB_OFFLINE"] = "1"
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    state = offline_state()
    if not (state["huggingface_hub_is_offline_mode"] and state["transformers_is_offline_mode"]):
        raise SystemExit(f"could not switch the Hugging Face libraries to offline mode: {state}")
    return {
        **state,
        "method": (
            "HF_HUB_OFFLINE=1 in the environment and huggingface_hub.constants.HF_HUB_OFFLINE set "
            "to True before any model is loaded (the constant is read when huggingface_hub is "
            "imported, so the variable alone has no effect after import); every Hub request then "
            "raises and files resolve from the local cache only"
        ),
    }


def offline_environment() -> dict[str, str]:
    return {**os.environ, "HF_HUB_OFFLINE": "1"}


def pinned_weights_file(name: str, revision: str) -> Record:
    for filename in WEIGHT_FILES:
        path = try_to_load_from_cache(name, filename, revision=revision)
        if not isinstance(path, str):
            continue
        resolved = Path(path)
        snapshot = resolved.parent.name
        return {
            "file": filename,
            "path": str(resolved),
            "snapshot": snapshot,
            "snapshot_is_pinned_revision": snapshot == revision,
            "blob": resolved.resolve().name,
            "format": "safetensors" if filename.startswith("model.safetensors") else "pytorch_bin",
        }
    raise SystemExit(f"{name}@{revision}: no weights file in the local cache among {WEIGHT_FILES}")
