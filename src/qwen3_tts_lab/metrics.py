"""Local process/device measurements shared by both benchmark backends."""
from __future__ import annotations

import math
import statistics
import subprocess
import threading
import time

import psutil


class ResourceMonitor:
    """Sample process-tree RSS and total GPU-0 memory (not PyTorch allocator usage)."""

    def __init__(self, pid: int, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        started = time.monotonic()
        while not self.stop_event.is_set():
            sample = {"elapsed_s": time.monotonic() - started}
            try:
                process = psutil.Process(self.pid)
                sample["rss_bytes"] = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
            except (psutil.Error, OSError):
                pass
            try:
                result = subprocess.run([
                    "nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                ], capture_output=True, text=True, timeout=3, check=True)
                sample["gpu_used_mib"] = int(result.stdout.strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                if not self.errors:
                    self.errors.append("GPU memory query unavailable; device peak is unverified.")
            self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=4)
        gpu = [s["gpu_used_mib"] for s in self.samples if "gpu_used_mib" in s]
        rss = [s["rss_bytes"] for s in self.samples if "rss_bytes" in s]
        return {
            "peak_device_used_mib": max(gpu) if gpu else None,
            "peak_process_tree_rss_bytes": max(rss) if rss else None,
            "resource_sample_count": len(self.samples),
            "resource_errors": self.errors,
            "resource_samples": self.samples,
            "memory_method": "0.5s sampling: GPU-0 total device memory, process-tree RSS; peaks may be missed; WSL process attribution limited",
        }


def real_time_factor(seconds: float, audio_seconds: float) -> float:
    if not all(math.isfinite(v) for v in (seconds, audio_seconds)) or seconds < 0 or audio_seconds <= 0:
        raise ValueError("RTF requires finite non-negative time and positive audio duration.")
    return seconds / audio_seconds


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["backend"], row["conditioning"], row["passage"]), []).append(row)
    result = []
    for (backend, mode, passage), group in groups.items():
        entry = {"backend": backend, "conditioning": mode, "passage": passage,
                 "runs": len(group), "failures": sum(r["status"] != "ok" for r in group)}
        for metric in ("process_rtf", "synthesis_rtf", "peak_device_used_mib", "peak_process_tree_rss_bytes"):
            values = [r[metric] for r in group if r["status"] == "ok" and r.get(metric) is not None]
            entry[metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
        result.append(entry)
    return result
