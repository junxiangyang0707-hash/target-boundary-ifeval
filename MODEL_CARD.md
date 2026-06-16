# Model Card

## Task

The models in this repository are prompt-only boundary predictors. Given a
prompt `x`, they predict whether a fixed local target model `T` will produce an
answer that passes a deterministic instruction-following checker.

They do not generate answers and are not instruction-following models.

## Model Families

- M1 TF-IDF: raw-prompt byte-BPE token n-gram TF-IDF plus logistic regression.
- M3 mean: small supervised bidirectional Transformer encoder with mean
  pooling and an MLP head.
- M4 frozen: instruction-following-domain pretrained encoder, frozen, with a
  target-specific MLP head.

## Intended Use

These models are useful for analyzing and ranking prompts by predicted
pass/fail behavior of a fixed target model under deterministic checks.
Potential uses include prompt triage, evaluation-set construction, and
diagnostic analysis of target-model brittleness.

## Not Intended For

- Generating responses.
- Replacing full instruction-following evaluation.
- Reporting a standard IFEval leaderboard score.
- Interpreting predicted scores as calibrated probabilities without additional
  calibration.

## Limitations

The included results use one target model, one task family, and a deterministic
checker. The atomic held-out test set was inspected during development, so the
headline results should be treated as evidence for boundary compressibility,
not as a final benchmark.
