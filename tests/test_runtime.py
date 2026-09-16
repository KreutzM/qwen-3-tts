from qwen3_tts_lab.runtime import DEFAULT_MODEL_ID, preferred_attention_backend


def test_default_model_is_voice_clone_base() -> None:
    assert DEFAULT_MODEL_ID == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def test_attention_backend_is_supported_value() -> None:
    assert preferred_attention_backend() in {"sdpa", "flash_attention_2"}
