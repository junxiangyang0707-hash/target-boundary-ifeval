# Target Boundary Prediction for IFEval-Style Checks

This repository is a reproducible experiment package and technical-blog
material set for a small boundary-model study.

The practical question is:

> Can a small prompt-only model predict whether a fixed local LLM will pass or
> fail deterministic, IFEval-style instruction-following checks before running
> the target model?

The answer in this experiment is yes, with important caveats. On the strict
atomic held-out split, the positive-rate AUPRC baseline is 14.8%, M1 TF-IDF
reaches 36.6%, and a tiny supervised bidirectional Transformer reaches
46.1% ± 7.8% AUPRC.

## What This Repo Is

This repository is intended as:

- a technical blog material package
- a lightweight reproducibility artifact for the blog figures
- an auditable record of split/tokenizer/result tables
- source code and configs for rerunning the experiment when local data and
  compute are available

It is not an IFEval leaderboard submission. It predicts the pass/fail boundary
of one fixed target model, Qwen3-4B-Instruct-2507, under deterministic checks.

## Main Results

| Model/config | Test AUPRC |
|---|---:|
| Positive-rate baseline | 14.8% |
| M1 TF-IDF full | 36.6% |
| M3 mean 40k | 43.7% ± 3.2% |
| M3 mean full | 46.1% ± 7.8% |
| M4 frozen 40k | 41.7% ± 5.2% |
| M4 frozen full | 39.8% ± 2.3% |

![Strict atomic-tokenizer multiseed test results](docs/assets/figure4_strict_main_result.png)

## Repository Layout

```text
docs/
  boundary_prediction_en.md
  boundary_prediction_zh.md
  assets/

results/
  tables/
  predictions/

data/
  splits/

boundary-if/
  src/
  scripts/
  configs/
  tests/
  Dockerfile
  docker-compose*.yml
  pyproject.toml
```

## Versioned Scope

v0.1 scope:

- English and Chinese blog drafts.
- Blog figures.
- Attribution and citation metadata.

v0.2 scope:

- v0.1 materials.
- Aggregate metrics tables.
- Validation/test predictions used by blog analyses.
- Split manifests.
- Code, configs, tests, and Docker files.

Excluded by design:

- raw promptsets
- raw target-model responses
- local model checkpoints
- target model weights
- Hugging Face/vLLM caches
- W&B run directories
- local chat logs and development notes

## Included Results

Core tables are in `results/tables/`:

- `run_metrics.csv`
- `run_metrics_by_config.csv`
- `selection_table.csv`
- `split_summary.csv`
- `tokenizer_audit.csv`
- `selective_metrics.csv`
- `topk_metrics.csv`
- `per_constraint_metrics.csv`
- `feature_coefficients.csv`

Predictions used for the final blog analyses are in:

```text
results/predictions/strict_atomic_blog_predictions.parquet
```

Split manifests are in:

```text
data/splits/
```

These artifacts do not include prompt text or target model raw outputs.

## Running the Code

The original workflow is Docker-first:

```bash
cd boundary-if
docker compose build app
docker compose run --rm app pytest
```

For GPU experiments:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm app pytest
```

Full target-model relabeling requires local model weights and a vLLM-compatible
GPU environment. The lightweight published artifacts are sufficient to inspect
the reported metrics and figures without rerunning target-model inference.

## Blog

- English: [`docs/boundary_prediction_en.md`](docs/boundary_prediction_en.md)
- Chinese: [`docs/boundary_prediction_zh.md`](docs/boundary_prediction_zh.md)

## Attribution

See [`ATTRIBUTION.md`](ATTRIBUTION.md). This work uses IFEval-style tasks and
checker logic as instruction/checker sources for target-specific boundary
labels. It does not report a standard IFEval benchmark score.

## License

Code in this repository is released under Apache-2.0 unless otherwise noted.
Blog text, figures, and tabular research artifacts may be reused with
attribution under CC BY 4.0. Vendored IFEval checker files retain original
Google Research Apache-2.0 headers.
