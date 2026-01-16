from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlmin.buffer import BufferReader
from rlmin.dataset import load_prompts_from_cfg


@dataclass
class TrainState:
    step: int = 0


def setup_distributed(cfg: DictConfig) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cfg.training.world_size > 1 or world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return rank, local_rank


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def build_model(cfg: DictConfig, device: torch.device) -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    model.to(device)

    if cfg.lora.enabled:
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[device.index])

    return model, tokenizer


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def compute_logprob(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    device: torch.device,
) -> torch.Tensor:
    full_text = prompt + response
    enc_full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)

    input_ids = enc_full.input_ids.to(device)
    attention_mask = enc_full.attention_mask.to(device)

    labels = input_ids.clone()
    prompt_len = enc_prompt.input_ids.shape[1]
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    token_count = (labels != -100).sum().clamp(min=1)
    logprob = -loss * token_count
    return logprob


def run_eval(cfg: DictConfig, model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device) -> dict:
    dataset = load_prompts_from_cfg(cfg.eval)
    if cfg.eval.max_samples:
        dataset = dataset[: cfg.eval.max_samples]

    generator = unwrap_model(model)
    generator.eval()
    total = 0
    correct = 0

    for i in range(0, len(dataset), cfg.eval.batch_size):
        batch = dataset[i : i + cfg.eval.batch_size]
        prompts = [item.prompt for item in batch]
        answers = [item.answer for item in batch]
        enc = tokenizer(prompts, return_tensors=\"pt\", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = generator.generate(
                **enc,
                max_new_tokens=cfg.eval.max_new_tokens,
                temperature=cfg.eval.temperature,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [text[len(prompt) :] for text, prompt in zip(decoded, prompts)]

        for response, answer in zip(responses, answers):
            if answer is None:
                continue
            total += 1
            if response.strip() == answer.strip():
                correct += 1

    accuracy = correct / total if total else 0.0
    return {\"eval/accuracy\": accuracy, \"eval/total\": total}


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    rank, local_rank = setup_distributed(cfg)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if cfg.wandb.enabled and is_main_process():
        import wandb

        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg))

    model, tokenizer = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    buffer = BufferReader(Path(cfg.buffer.dir), poll_interval_s=cfg.buffer.poll_interval_s)
    state = TrainState()

    for item in buffer.iter_items():
        if cfg.training.max_steps and state.step >= cfg.training.max_steps:
            break
        prompt = item["prompt"]
        response = item["response"]
        reward = float(item["reward"])

        model.train()
        logprob = compute_logprob(model, tokenizer, prompt, response, device)
        loss = -(logprob * reward * cfg.training.reward_scale)

        loss.backward()
        if (state.step + 1) % cfg.training.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if cfg.wandb.enabled and is_main_process() and state.step % cfg.wandb.log_every == 0:
            import wandb

            wandb.log({"loss": loss.item(), "reward": reward, "step": state.step})

        if cfg.eval.enabled and is_main_process() and state.step % cfg.eval.every_steps == 0:
            metrics = run_eval(cfg, model, tokenizer, device)
            if cfg.wandb.enabled:
                import wandb

                wandb.log({**metrics, "step": state.step})
            print(json.dumps({"step": state.step, **metrics}))

        if is_main_process() and state.step % cfg.training.print_every == 0:
            print(json.dumps({"step": state.step, "loss": loss.item(), "reward": reward}))

        state.step += 1


if __name__ == "__main__":
    main()
