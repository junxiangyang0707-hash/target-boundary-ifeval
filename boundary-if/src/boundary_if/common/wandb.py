from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _none_if_empty(value: Any) -> Any:
    return None if value in ("", "null", "None") else value


def init_wandb(cfg: DictConfig, job_type: str | None = None):
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None

    import wandb

    wandb_dir = Path(str(wandb_cfg.get("dir", "runs/wandb")))
    wandb_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    return wandb.init(
        project=str(wandb_cfg.get("project")),
        entity=_none_if_empty(wandb_cfg.get("entity")),
        mode=str(wandb_cfg.get("mode", "online")),
        dir=str(wandb_dir),
        group=_none_if_empty(wandb_cfg.get("group")),
        name=_none_if_empty(wandb_cfg.get("name")),
        job_type=job_type or str(wandb_cfg.get("job_type", "run")),
        tags=list(wandb_cfg.get("tags", [])),
        config=resolved_cfg,
    )
