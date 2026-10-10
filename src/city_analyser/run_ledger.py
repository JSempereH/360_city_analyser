"""Append-only record of what a batch run has finished, so reruns never repeat work.

Validating an analysis from scratch means hashing the panorama and masks and
re-reading the manifest. For a city that is hours before the first new
panorama. The ledger answers "already done?" from one small file: an entry
holds the settings fingerprint and the panorama's size and modification time,
so a changed setting or a replaced image is redone and nothing else is.
Entries are one JSON line each, flushed and synced when written, so a crash
or a Colab disconnect loses at most the line being written.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .analysis_cache import canonical_json_sha256

LEDGER_NAME = "batch-ledger.jsonl"
DONE, FAILED, SKIPPED = "done", "failed", "skipped"
# A transient failure is retried by later runs until it has failed this many times.
MAX_TOTAL_ATTEMPTS = 6


def ledger_path(data_root: Path, dataset_id: str) -> Path:
    return data_root / dataset_id / LEDGER_NAME


def settings_fingerprint(stage: str, configuration: dict[str, Any]) -> str:
    """``stage`` is "analysis" or "perception"; ``configuration`` the settings that stage depends on."""
    return canonical_json_sha256({"stage": stage, "configuration": configuration})


def file_signature(path: Path) -> dict[str, int]:
    status = path.stat()
    return {"size": status.st_size, "mtime_ns": status.st_mtime_ns}


class RunLedger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a crash
            if isinstance(entry, dict) and isinstance(entry.get("image"), str) and isinstance(entry.get("stage"), str):
                self.entries[f"{entry['stage']}:{entry['image']}"] = entry

    def get(self, stage: str, image_id: str) -> dict[str, Any] | None:
        return self.entries.get(f"{stage}:{image_id}")

    def is_done(self, stage: str, image_id: str, fingerprint: str, signature: dict[str, int]) -> bool:
        entry = self.get(stage, image_id)
        return (
            entry is not None and entry.get("status") == DONE and entry.get("settings") == fingerprint
            and entry.get("size") == signature["size"] and entry.get("mtime_ns") == signature["mtime_ns"]
        )

    def total_failures(self, stage: str, image_id: str) -> int:
        entry = self.get(stage, image_id)
        return int(entry.get("attempts", 0)) if entry is not None and entry.get("status") == FAILED else 0

    def record(self, stage: str, image_id: str, status: str, fingerprint: str, signature: dict[str, int] | None, **details: Any) -> None:
        entry = {
            "image": image_id, "stage": stage, "status": status, "settings": fingerprint,
            **(signature or {}), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details,
        }
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as output:
                output.write(line)
                output.flush()
                os.fsync(output.fileno())
            self.entries[f"{stage}:{image_id}"] = entry
