"""Download pinned artifacts with integrity checks, without model dependencies."""
from __future__ import annotations

import argparse
import hashlib
import fcntl
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest must contain a non-empty files list.")
    names: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each artifact must be a JSON object.")
        name = item.get("filename", "")
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError("Artifact filenames must be unique basenames.")
        names.add(name)
        digest = item.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for {name}.")
        if type(item.get("size")) is not int or item["size"] <= 0:
            raise ValueError(f"Invalid size for {name}.")
        url = item.get("url", "")
        if not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).netloc:
            raise ValueError(f"HTTPS URL required for {name}.")
    return manifest


def matches(path: Path, item: dict) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == item["sha256"]


def retrieve(manifest: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".download.lock"
    if lock.is_symlink():
        raise ValueError("Refusing a download-lock symlink.")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        _retrieve_files(manifest, destination)


def _retrieve_files(manifest: dict, destination: Path) -> None:
    for item in manifest["files"]:
        target = destination / item["filename"]
        if target.exists() or target.is_symlink():
            if not matches(target, item):
                raise ValueError(f"Unexpected artifact at {target}; move it aside before retrying.")
            print(f"Verified existing {target.name}", flush=True)
            continue
        partial = target.with_name(target.name + ".partial")
        if partial.is_symlink():
            raise ValueError(f"Refusing partial symlink: {partial}")
        # curl resumes interrupted transfers; no credentials are used for these public files.
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
            "--max-time", "3600", "--continue-at", "-", "--output", str(partial), item["url"],
        ], check=True)
        if not matches(partial, item):
            raise ValueError(f"Checksum/size mismatch at {partial}; move it aside before retrying.")
        # Hard-link publication refuses a concurrent existing target instead of overwriting it.
        target.hardlink_to(partial)
        partial.unlink()
        print(f"Downloaded and verified {target.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        retrieve(read_manifest(args.manifest), args.destination)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Artifact download failed: {exc}\n")


if __name__ == "__main__":
    main()
