# Data Card

## Scope

This repository includes lightweight artifacts for auditing and reproducing
the blog figures:

- split manifests under `data/splits/`
- aggregate result tables under `results/tables/`
- validation/test predictions under `results/predictions/`
- blog figures under `docs/assets/`

It intentionally does not include the full normalized promptset, raw target
model responses, local model weights, Hugging Face cache, W&B runs, or local
experiment directories.

## Included v0.2 Artifacts

`results/predictions/strict_atomic_blog_predictions.parquet` contains:

- `prompt_id`
- model/run metadata
- split name
- deterministic checker label
- predicted probability
- token-length, cluster, and constraint-count metadata

It does not contain prompt text or target model raw outputs.

`data/splits/*.parquet` contains split assignments and constraint metadata such
as `prompt_id`, `base_key`, `instruction_ids`, `constraint_signature`, and
`constraint_family_signature`. These files do not contain prompt text or target
model raw outputs.

## Source and License Notes

The task format is IFEval-style and uses automatically checkable instruction
constraints. The original IFEval data is CC BY 4.0, and Google Research source
code is Apache 2.0. This repository uses those materials as instruction/checker
sources for constructing target-specific boundary labels.

Downstream users who rebuild the full promptset or target-model responses
should verify the licenses of any upstream source datasets and target model
outputs they choose to redistribute.
