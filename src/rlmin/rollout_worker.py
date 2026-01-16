from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams

from rlmin.buffer import BufferWriter
from rlmin.rewards import LLMReward, VerifiableReward


@dataclass
class PromptItem:
    prompt: str
    answer: str | None = None


def load_prompts(path: Path) -> list[PromptItem]:
    prompts: list[PromptItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append(PromptItem(prompt=item["prompt"], answer=item.get("answer")))
    return prompts


def build_reward(cfg: DictConfig, reward_llm: LLM | None) -> tuple[str, object]:
    if cfg.reward.type == "verifiable":
        return "verifiable", VerifiableReward(normalize=cfg.reward.normalize)
    if cfg.reward.type == "llm":
        if reward_llm is None:
            raise ValueError("reward.llm requires reward model")
        return "llm", LLMReward(
            llm=reward_llm,
            prompt_template=cfg.reward.prompt_template,
            max_tokens=cfg.reward.max_tokens,
        )
    raise ValueError(f"Unknown reward type: {cfg.reward.type}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    prompt_items = load_prompts(Path(cfg.data.prompts_path))
    buffer = BufferWriter(Path(cfg.buffer.dir))

    llm = LLM(
        model=cfg.rollout.model,
        tensor_parallel_size=cfg.rollout.tensor_parallel_size,
        gpu_memory_utilization=cfg.rollout.gpu_memory_utilization,
    )
    reward_llm = None
    if cfg.reward.type == "llm":
        reward_llm = LLM(
            model=cfg.reward.model,
            tensor_parallel_size=cfg.reward.tensor_parallel_size,
            gpu_memory_utilization=cfg.reward.gpu_memory_utilization,
        )

    reward_kind, reward_fn = build_reward(cfg, reward_llm)
    params = SamplingParams(
        max_tokens=cfg.rollout.max_tokens,
        temperature=cfg.rollout.temperature,
        top_p=cfg.rollout.top_p,
    )

    idx = 0
    while True:
        batch = []
        prompts: list[str] = []
        answers: list[str | None] = []
        for _ in range(cfg.rollout.batch_size):
            item = prompt_items[idx % len(prompt_items)]
            idx += 1
            prompts.append(item.prompt)
            answers.append(item.answer)
        outputs = llm.generate(prompts, params)
        responses = [out.outputs[0].text for out in outputs]

        if reward_kind == "verifiable":
            rewards = reward_fn.score(prompts, responses, answers)
        else:
            rewards = reward_fn.score(prompts, responses)

        for prompt, response, reward, answer in zip(prompts, responses, rewards, answers):
            batch.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "reward": float(reward),
                    "answer": answer,
                }
            )
        buffer.write_batch(batch, prefix="rollout")


if __name__ == "__main__":
    main()
