"""Bounded local llama-tts execution; raw metadata stays in ignored outputs."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .metrics import ResourceMonitor, real_time_factor

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/qwen3-tts-llama"


@dataclass(frozen=True)
class RunConfig:
    text: str
    executable: Path = ROOT / ".tools/llama.cpp/build/bin/llama-tts"
    model: Path = MODEL_DIR / "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
    mmproj: Path = MODEL_DIR / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
    reference: Path | None = None
    language: str = "de"
    frames: int = 300
    timeout: float = 300
    gpu_layers: int = 99
    context: int = 4096
    attention: str = "off"
    seed: int = 42
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    outputs: Path = ROOT / "outputs/llama-cpp"


class RunError(RuntimeError):
    def __init__(self, message: str, returncode: int, result_dir: Path):
        super().__init__(message)
        self.returncode = returncode
        self.result_dir = result_dir


def validate(config: RunConfig) -> None:
    if not config.text.strip() or "\0" in config.text:
        raise ValueError("Provide non-empty text without NUL bytes.")
    for label, path in (("Executable", config.executable), ("Talker model", config.model), ("Audio companion", config.mmproj)):
        if not path.is_file():
            raise ValueError(f"{label} missing: {path}. Run the bootstrap/model-download scripts first.")
    if not os.access(config.executable, os.X_OK):
        raise ValueError("llama-tts executable is not executable.")
    if config.reference is not None and not config.reference.is_file():
        raise ValueError("Reference audio does not exist; use a consented local file.")
    if config.reference is not None and not config.reference.resolve().is_relative_to((ROOT / "local_data").resolve()):
        raise ValueError("Keep consented reference recordings inside ignored local_data/.")
    if config.language not in {"de", "en", "zh", "it", "pt", "es", "ja", "ko", "fr", "ru"}:
        raise ValueError("Unsupported language code.")
    if config.context < 256 or config.frames >= config.context or config.attention not in {"off", "on", "auto"}:
        raise ValueError("Context must be >=256 and exceed the frame cap; attention must be off/on/auto.")
    if config.frames <= 0 or config.gpu_layers < 0 or config.top_k <= 0 or not 0 <= config.seed <= 0xFFFFFFFF:
        raise ValueError("Frames/top-k must be positive; GPU layers non-negative; seed must fit uint32.")
    if not math.isfinite(config.timeout) or config.timeout <= 0:
        raise ValueError("Timeout must be finite and positive.")
    if not math.isfinite(config.temperature) or config.temperature < 0 or not math.isfinite(config.top_p) or not 0 < config.top_p <= 1:
        raise ValueError("Temperature must be finite/non-negative and top-p in (0, 1].")
    output_root = (ROOT / "outputs").resolve()
    if not config.outputs.resolve().is_relative_to(output_root):
        raise ValueError("Output directory must be inside the project's ignored outputs/.")


def command(config: RunConfig, output: Path) -> list[str]:
    args = [str(config.executable.resolve()), "-m", str(config.model.resolve()),
            "--mmproj", str(config.mmproj.resolve()), "--tts-lang", config.language,
            "-ngl", str(config.gpu_layers), "-c", str(config.context), "--flash-attn", config.attention,
            "--verbosity", "4", "--offline", "-n", str(config.frames), "--seed", str(config.seed),
            "--temp", str(config.temperature), "--top-k", str(config.top_k), "--top-p", str(config.top_p),
            "-p", config.text, "--output", str(output)]
    if config.reference:
        args.extend(["--tts-speaker-file", str(config.reference.resolve())])
    return args


def audio_info(path: Path) -> dict:
    import numpy as np
    import soundfile as sf
    data, rate = sf.read(path, always_2d=True)
    if rate <= 0 or not len(data) or not np.isfinite(data).all():
        raise ValueError("Output audio is empty or contains invalid samples.")
    rms = float(np.sqrt(np.mean(data ** 2)))
    if rms < 1e-7:
        raise ValueError("Output audio is silent.")
    return {"sample_rate": rate, "audio_duration_s": len(data) / rate, "audio_rms": rms}


def parse_timings(log: str) -> dict:
    match = re.search(r"timings: prompt eval ([\d.]+)s \+ generation ([\d.]+)s \+ vocoder ([\d.]+)s = total ([\d.]+)s", log)
    result = {}
    if match:
        result.update(zip(("prompt_eval_s", "generation_s", "vocoder_s", "synthesis_s"), map(float, match.groups())))
    match = re.search(r"generated (\d+) frames,", log)
    if match:
        result["generated_frames"] = int(match.group(1))
    return result


def provenance(config: RunConfig) -> dict:
    """Report pins only for managed artifacts; custom paths have unknown provenance."""
    result = {"code_revision": None, "model_revision": None, "model_id": None, "quantization": "custom"}
    pin = ROOT / "config/llama_cpp_revision.txt"
    if config.executable.resolve() == (ROOT / ".tools/llama.cpp/build/bin/llama-tts").resolve():
        if not pin.is_file():
            raise ValueError("Missing code revision pin; restore config/llama_cpp_revision.txt.")
        expected = pin.read_text().strip()
        try:
            git = subprocess.run(["git", "-C", str(ROOT / ".tools/llama.cpp"), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Cannot verify llama.cpp checkout; inspect it and rerun bootstrap.") from exc
        if git.returncode or git.stdout.strip() != expected:
            raise ValueError("llama.cpp checkout differs from the pin; rerun bootstrap before synthesis.")
        result["code_revision"] = expected
    model_dir = ROOT / "models/qwen3-tts-llama"
    if (config.model.resolve() == (model_dir / "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf").resolve()
            and config.mmproj.resolve() == (model_dir / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf").resolve()):
        metadata = json.loads((ROOT / "config/llama_model.json").read_text())
        result.update(model_revision=metadata["revision"], model_id=metadata["source_model"], quantization="Q8_0")
    return result


def execute(config: RunConfig) -> Path:
    validate(config)
    identity = provenance(config)
    config.outputs.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=config.outputs)).resolve()
    output = directory / "speech.wav"
    args = command(config, output)
    record = {"backend": "llama.cpp", "conditioning": "speaker_only" if config.reference else "none",
              "input_characters": len(config.text), "language": config.language,
              "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "quantization": "Q8_0" if "Q8_0" in config.model.name else "custom",
              "frame_limit": config.frames, "context": config.context, "talker_attention": config.attention, "sampling": {"seed": config.seed, "temperature": config.temperature, "top_k": config.top_k, "top_p": config.top_p},
              "command": args, "status": "error", "listening_assessment": "pending"} | identity
    start = time.monotonic()
    returncode = 1
    error: str | None = None
    try:
        with (directory / "process.log").open("w") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            monitor = ResourceMonitor(process.pid)
            monitor.start()
            try:
                returncode = process.wait(timeout=config.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = 124
                error = f"Synthesis exceeded {config.timeout}s; inspect process.log."
            finally:
                record["process_wall_s"] = time.monotonic() - start
                record.update(monitor.finish())
        if returncode != 0:
            error = error or f"llama-tts failed with exit status {returncode}; inspect process.log."
        else:
            log_text = (directory / "process.log").read_text(errors="replace")
            record.update(parse_timings(log_text))
            record.update(audio_info(output))
            if record.get("generated_frames", config.frames) >= config.frames:
                error = "Output reached the frame limit or frame count is unverified; increase the bound and inspect termination."
                returncode = 1
            else:
                record["status"] = "ok"
                record["process_rtf"] = real_time_factor(record["process_wall_s"], record["audio_duration_s"])
                record["synthesis_rtf"] = real_time_factor(record["synthesis_s"], record["audio_duration_s"]) if record.get("synthesis_s") else None
    except (OSError, ValueError, RuntimeError) as exc:
        error = str(exc)
        returncode = 1
    finally:
        record["returncode"] = returncode
        record["error"] = error
        (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if error:
        raise RunError(error, returncode, directory)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    parser.add_argument("--reference", type=Path)
    for name in ("executable", "model", "mmproj", "outputs"):
        parser.add_argument(f"--{name}", type=Path, default=getattr(RunConfig(""), name))
    for name, kind in (("frames", int), ("context", int), ("timeout", float), ("gpu-layers", int), ("seed", int), ("temperature", float), ("top-k", int), ("top-p", float)):
        parser.add_argument(f"--{name}", type=kind, default=getattr(RunConfig(""), name.replace("-", "_")))
    parser.add_argument("--language", default="de")
    parser.add_argument("--attention", choices=("off", "on", "auto"), default="off")
    args = vars(parser.parse_args())
    try:
        file = args.pop("text_file")
        if file:
            args["text"] = file.read_text(encoding="utf-8").strip()
        directory = execute(RunConfig(**args))
    except RunError as exc:
        parser.exit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode, f"{exc} Results: {exc.result_dir}\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(directory)


if __name__ == "__main__":
    main()
