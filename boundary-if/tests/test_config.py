from __future__ import annotations

from omegaconf import OmegaConf

from boundary_if.common.config import load_config


def test_default_config_loads() -> None:
    cfg = load_config()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["data"]["hf_path"] == "allenai/IF_multi_constraints_upto5"
    assert resolved["wandb"]["project"] == "boundary-if"
    assert resolved["train"]["label_name"] == "loose_pass"
