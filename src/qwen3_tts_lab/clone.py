from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from .runtime import DEFAULT_MODEL_ID, AttentionBackend, load_model


def _read_text(value: str | None, file_path: Path | None, label: str) -> str:
    if value and file_path:
        raise ValueError(f"Use either --{label} or --{label}-file, not both.")
    if file_path:
        text = file_path.read_text(encoding="utf-8").strip()
    else:
        text = (value or "").strip()
    if not text:
        raise ValueError(f"A non-empty {label.replace('-', ' ')} is required.")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a German Qwen3-TTS voice-clone sample.")
    parser.add_argument("--ref-audio", type=Path, required=True, help="User-owned reference WAV/MP3 file.")
    parser.add_argument("--ref-text", help="Exact transcript of the reference audio.")
    parser.add_argument("--ref-text-file", type=Path, help="UTF-8 file containing the exact reference transcript.")
    parser.add_argument("--text", help="Text to synthesize.")
    parser.add_argument("--text-file", type=Path, help="UTF-8 file containing text to synthesize.")
    parser.add_argument("--out", type=Path, required=True, help="Output WAV path, normally under outputs/.")
    parser.add_argument("--language", default="German")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--attention",
        choices=("auto", "sdpa", "flash_attention_2"),
        default="auto",
        help="Attention backend. auto uses FlashAttention only when installed, otherwise SDPA.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.ref_audio.is_file():
        raise SystemExit(f"Reference audio not found: {args.ref_audio}")

    try:
        ref_text = _read_text(args.ref_text, args.ref_text_file, "ref-text")
        target_text = _read_text(args.text, args.text_file, "text")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    model = load_model(args.model, args.attention)  # type: ignore[arg-type]
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(args.ref_audio),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    wavs, sample_rate = model.generate_voice_clone(
        text=target_text,
        language=args.language,
        voice_clone_prompt=prompt,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wavs[0], sample_rate)
    print(f"Wrote {args.out} ({sample_rate} Hz)")


if __name__ == "__main__":
    main()
