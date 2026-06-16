from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return project_root() / "configs"


def load_config(config_name: str = "config", overrides: Sequence[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir())):
        return compose(config_name=config_name, overrides=list(overrides or []))
