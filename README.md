# llm-rl-minimal

Minimal RL training skeleton for LLMs with async rollouts via vLLM, PyTorch training, hydra configs, and wandb logging. It supports collocated or split GPU layouts, and LoRA or full-parameter fine-tuning.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Run (single GPU, collocated)
```bash
rlmin-launcher
```

### Run (split GPUs)
```bash
rlmin-launcher gpus.mode=split gpus.gen_gpus=[0] gpus.train_gpus=[1]
```

### Run (multi-GPU training)
```bash
rlmin-launcher gpus.mode=split gpus.gen_gpus=[0] gpus.train_gpus=[1,2,3] training.world_size=3
```

### Run (Qwen3-1.7B on Math500, eval on AIME24)
```bash
rlmin-launcher data=math500 eval=aime24 model=qwen3_1_7b rollout=qwen3_1_7b
```

### Run (HF datasets: Math500 + AIME2024)
```bash
rlmin-launcher data=hf_math500 eval=hf_aime2024 model=qwen3_1_7b rollout=qwen3_1_7b
```

## Notes
- Async training uses a file-backed buffer in `runs/buffer` by default.
- vLLM is used for rollouts; training uses `transformers` + PyTorch.
- Reward can be verifiable (exact match) or LLM-based (scalar score prompt).
