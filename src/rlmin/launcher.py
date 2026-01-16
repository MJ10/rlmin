from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf


def _format_cuda_visible(gpus: list[int]) -> str:
    return ",".join(str(gpu) for gpu in gpus)


def _launch_rollout(cfg: DictConfig, env: dict) -> subprocess.Popen:
    cmd = ["python", "-m", "rlmin.rollout_worker", "hydra.run.dir=."]
    return subprocess.Popen(cmd, env=env)


def _launch_trainer(cfg: DictConfig, env: dict) -> subprocess.Popen:
    if cfg.training.world_size > 1:
        cmd = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={cfg.training.world_size}",
            "-m",
            "rlmin.trainer",
            "hydra.run.dir=.",
        ]
    else:
        cmd = ["python", "-m", "rlmin.trainer", "hydra.run.dir=."]
    return subprocess.Popen(cmd, env=env)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    base_env = os.environ.copy()
    base_env["HYDRA_FULL_ERROR"] = "1"

    if cfg.gpus.mode == "split":
        gen_env = base_env.copy()
        train_env = base_env.copy()
        gen_env["CUDA_VISIBLE_DEVICES"] = _format_cuda_visible(cfg.gpus.gen_gpus)
        train_env["CUDA_VISIBLE_DEVICES"] = _format_cuda_visible(cfg.gpus.train_gpus)
    else:
        gen_env = base_env.copy()
        train_env = base_env.copy()
        if cfg.gpus.all_gpus:
            vis = _format_cuda_visible(cfg.gpus.all_gpus)
            gen_env["CUDA_VISIBLE_DEVICES"] = vis
            train_env["CUDA_VISIBLE_DEVICES"] = vis

    rollout_proc = _launch_rollout(cfg, gen_env)
    trainer_proc = _launch_trainer(cfg, train_env)

    trainer_proc.wait()
    rollout_proc.terminate()
    rollout_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
