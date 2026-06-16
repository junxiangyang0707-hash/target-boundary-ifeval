from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from boundary_if.common.config import load_config
from boundary_if.common.wandb import init_wandb


def _write_smoke_dataset(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_dict(
        {
            "prompt_id": ["smoke-0001"],
            "user_prompt": ["Return a JSON object with one key named ok."],
            "loose_pass": [1],
        }
    )
    output_path = output_dir / "smoke_dataset.parquet"
    dataset.to_parquet(str(output_path))

    table = pa.table({"check": ["arrow"], "rows": [dataset.num_rows]})
    assert table.num_rows == 1
    return output_path


def run(cfg: DictConfig) -> None:
    print("Resolved Hydra config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    output_path = _write_smoke_dataset(Path(cfg.paths.reports_dir))
    print(f"HF Datasets/Arrow smoke file: {output_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    run = init_wandb(cfg, job_type="smoke")
    if run is not None:
        run.log(
            {
                "smoke/rows": 1,
                "smoke/cuda_available": int(torch.cuda.is_available()),
            }
        )
        run.finish()


def main() -> None:
    cfg = load_config(overrides=sys.argv[1:])
    run(cfg)


if __name__ == "__main__":
    main()
