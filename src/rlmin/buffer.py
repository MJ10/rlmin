from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class BufferWriter:
    buffer_dir: Path

    def __post_init__(self) -> None:
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

    def write_batch(self, items: Iterable[dict], prefix: str = "rollout") -> Path:
        ts = int(time.time() * 1000)
        tmp_path = self.buffer_dir / f"{prefix}_{ts}_{os.getpid()}.tmp"
        final_path = self.buffer_dir / f"{prefix}_{ts}_{os.getpid()}.jsonl"
        with tmp_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        tmp_path.replace(final_path)
        return final_path


@dataclass
class BufferReader:
    buffer_dir: Path
    poll_interval_s: float = 1.0
    delete_after_read: bool = True

    def __post_init__(self) -> None:
        self.buffer_dir.mkdir(parents=True, exist_ok=True)

    def _claim_file(self, path: Path) -> Path | None:
        processing_path = path.with_suffix(path.suffix + ".processing")
        try:
            path.replace(processing_path)
        except FileNotFoundError:
            return None
        return processing_path

    def iter_items(self) -> Iterable[dict]:
        while True:
            files = sorted(self.buffer_dir.glob("*.jsonl"))
            if not files:
                time.sleep(self.poll_interval_s)
                continue
            for path in files:
                claimed = self._claim_file(path)
                if claimed is None:
                    continue
                with claimed.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        yield json.loads(line)
                if self.delete_after_read:
                    claimed.unlink(missing_ok=True)
                else:
                    claimed.replace(path)
