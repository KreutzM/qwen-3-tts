from __future__ import annotations

from importlib.util import find_spec
from typing import Literal

import torch
from qwen_tts import Qwen3TTSModel

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
AttentionBackend = Literal["auto", "sdpa", "flash_attention_2"]


def preferred_dtype() -> torch.dtype:
    """Return the preferred half precision dtype for the current CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def preferred_attention_backend() -> str:
    """Use FlashAttention only when its Python package is actually installed."""
    return "flash_attention_2" if find_spec("flash_attn") is not None else "sdpa"


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    attention: AttentionBackend = "auto",
) -> Qwen3TTSModel:
    """Load the configured Qwen3-TTS model onto the first CUDA GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run scripts/doctor.sh and verify the CUDA-enabled PyTorch build before loading Qwen3-TTS."
        )

    backend = preferred_attention_backend() if attention == "auto" else attention
    dtype = preferred_dtype()

    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=dtype,
        attn_implementation=backend,
    )
