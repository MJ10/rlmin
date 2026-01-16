from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass
class PromptItem:
    prompt: str
    answer: str | None = None


def load_jsonl_prompts(path: Path, prompt_field: str, answer_field: str | None) -> list[PromptItem]:
    items: list[PromptItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prompt = raw[prompt_field]
            answer = raw.get(answer_field) if answer_field else None
            items.append(PromptItem(prompt=prompt, answer=answer))
    return items


def load_hf_prompts(
    dataset_name: str,
    split: str,
    prompt_field: str,
    answer_field: str | None,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> list[PromptItem]:
    ds = load_dataset(dataset_name, split=split, revision=revision, cache_dir=cache_dir)
    items: list[PromptItem] = []
    for row in ds:
        prompt = row[prompt_field]
        answer = row.get(answer_field) if answer_field else None
        items.append(PromptItem(prompt=prompt, answer=answer))
    return items


def load_prompts_from_cfg(cfg: Any) -> list[PromptItem]:
    if cfg.source == "jsonl":
        return load_jsonl_prompts(Path(cfg.prompts_path), cfg.prompt_field, cfg.answer_field)
    if cfg.source == "hf":
        return load_hf_prompts(
            dataset_name=cfg.dataset_name,
            split=cfg.dataset_split,
            prompt_field=cfg.prompt_field,
            answer_field=cfg.answer_field,
            revision=cfg.dataset_revision,
            cache_dir=cfg.cache_dir,
        )
    raise ValueError(f"Unknown data.source: {cfg.source}")
