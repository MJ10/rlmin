from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vllm import LLM, SamplingParams


@dataclass
class VerifiableReward:
    normalize: bool = True

    def score(self, prompts: list[str], responses: list[str], answers: list[str | None]) -> list[float]:
        scores = []
        for response, answer in zip(responses, answers):
            if answer is None:
                scores.append(0.0)
                continue
            match = response.strip() == answer.strip()
            scores.append(1.0 if match else 0.0)
        if self.normalize:
            return [float(s) for s in scores]
        return scores


@dataclass
class LLMReward:
    llm: LLM
    prompt_template: str
    max_tokens: int = 4

    def score(self, prompts: list[str], responses: list[str]) -> list[float]:
        reward_prompts = [
            self.prompt_template.format(prompt=p, response=r) for p, r in zip(prompts, responses)
        ]
        params = SamplingParams(max_tokens=self.max_tokens, temperature=0.0)
        outputs = self.llm.generate(reward_prompts, params)
        scores: list[float] = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            try:
                scores.append(float(text.split()[0]))
            except ValueError:
                scores.append(0.0)
        return scores
