"""
A deliberately simple disk-backed cache. Its only job is to stop identical
LLM calls — retries, re-running the same demo question, the eval harness
re-scoring after a code change — from spending free-tier quota twice on the
same answer.
"""

import json
import time
from pathlib import Path
from typing import Optional


class DiskCache:
    def __init__(self, path: str = ".cache/argus_cache.json", ttl_seconds: int = 60 * 60 * 24 * 7):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._data = {}

    def get(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self.ttl:
            return None
        return entry["value"]

    def set(self, key: str, value: str) -> None:
        self._data[key] = {"value": value, "ts": time.time()}
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._data))


