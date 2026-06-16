# Predicting When a Local LLM Will Pass an Instruction Checker

This is a small boundary-model study. The practical question behind it is whether failures of a fixed local LLM on verifiable instructions are predictable before the model is even run. If they are, a small prompt-only model could be useful for prompt triage, evaluation-set construction, or for mapping where a target model is brittle.

The goal is not to imitate a target LLM's response. The goal is narrower:

> Given a prompt `x`, predict whether a fixed local target model `T` will produce an answer that passes a deterministic instruction-following checker.

The label is:

```text
y_T(x) = 1[checker(T(x)) = pass]
```

The target model in this run is a local Qwen3-4B-Instruct-2507 deployment. The task family is IFEval-style verifiable instructions: format constraints, keyword constraints, length constraints, punctuation constraints, and similar rules.

The central finding is that these target-model pass/fail boundaries are surprisingly compressible. On the strict atomic held-out split, the positive-rate baseline AUPRC is 14.8%. A token n-gram TF-IDF logistic regression baseline reaches 36.6%. A tiny supervised bidirectional Transformer reaches 46.1% ± 7.8% AUPRC.

Two clarifications are important for reading the result. First, the boundary model never sees the target answer at inference time; target answers are used only to create labels. Second, the ± values reported for neural models are mean ± standard deviation across training seeds in this run, not confidence intervals over test examples.

## Experimental Pipeline

The boundary model never sees the target answer at inference time. It only sees the prompt.

![Figure 1: Boundary prediction pipeline](assets/figure1_boundary_pipeline_revised.png)

*What to look for: The target answer is used only during labeling. At inference time, the boundary model receives the prompt and outputs a pass-likelihood score.*

The data pipeline is:

```text
Prompt
  -> local target LLM
  -> target response
  -> deterministic checker
  -> pass/fail label
  -> small boundary model
  -> P(pass)
```

This makes the problem different from model distillation. We are not training a smaller model to answer the prompt. We are training it to predict a behavioral boundary of the target model.

## Splits

The main benchmark uses an atomic constraint held-out split. Some atomic instruction IDs are withheld from train and validation and appear in test, so the test set asks whether the boundary model can generalize to unseen constraint types.

| Split | Train rows | Val rows | Test rows | Test positive rate | Baseline AUPRC | Held-out unit |
|---|---:|---:|---:|---:|---:|---|
| group-key split | 66,245 | 8,280 | 8,281 | 29.7% | 29.7% | `base_key` |
| atomic constraint held-out | 61,655 | 6,851 | 14,300 | 14.8% | 14.8% | `instruction_id` |
| composition C1 | 61,388 | 15,501 | 5,917 | 4.7% | 4.7% | `num_constraints` |
| composition C2 | 76,889 | 1,183 | 4,734 | 4.6% | 4.6% | `num_constraints` |

*What to look for: The headline split is the atomic constraint held-out row: 61,655 train, 6,851 validation, 14,300 test, and 14.8% positives in test.*

I also ran easier and alternative splits during development. The group-key split prevents original task leakage by grouping on `base_key`. Composition splits hold out higher numbers of simultaneous constraints. These are useful diagnostics, but the clean main result below uses only the strict atomic held-out protocol.

## Protocol Card

Here is the compact version of the protocol used for the headline result.

| Item | Value |
|---|---|
| Target model | Local Qwen3-4B-Instruct-2507 deployment |
| Task family | IFEval-style verifiable instructions |
| Label | `1[checker(T(x)) = pass]` |
| Boundary-model input at inference | Prompt text only |
| Main split | Atomic constraint held-out |
| Held-out unit | `instruction_id` |
| Main train / validation / test rows | 61,655 / 6,851 / 14,300 |
| Test positive rate | 14.8% |
| Random-ranker AUPRC baseline | 14.8% |
| Main tokenizer protocol | Raw-prompt tokenizer trained only on atomic-train prompts |
| Model selection | Validation-selected configs are separated from test-oracle diagnostics |
| Error bars | Mean ± standard deviation across training seeds |

The important detail is the held-out unit. Test prompts contain atomic instruction IDs that are absent from both training and validation, so the main number is not just measuring interpolation over familiar atomic constraints.

## Why AUPRC?

The strict atomic test set is class-imbalanced: only 14.8% of examples pass the checker. A model that mostly predicts failure can look decent under plain accuracy, even if it is bad at finding the prompts that will pass. AUPRC is a better fit for this study because the operational question is ranking: which prompts should be near the top if I want likely passes?

A random ranker has expected AUPRC equal to the positive rate. In this split, that is 14.8%. So the move from 14.8% to 36.6% or 46.1% is not a small cosmetic gain; it means the prompt-only model is putting substantially more true passes near the top of the ranked list.

## Models

I used three model families.

| Model | Description |
|---|---|
| M1 TF-IDF | Raw-prompt byte-BPE token n-gram TF-IDF plus logistic regression |
| M3 mean | Tiny supervised bidirectional Transformer encoder, mean pooling, MLP head |
| M4 frozen | IF-domain pretrained encoder, frozen, plus target-specific MLP head |

M1 is the lexical baseline: if it works, the boundary has strong surface-level regularities. M3 is the small supervised boundary model trained directly for this target model and checker. M4 is a representation-transfer baseline: the encoder is pretrained on instruction-following-style text and then frozen, so only a small target-specific head adapts to this pass/fail task.

This distinction matters for interpreting the result. If M4 wins, the main story is reusable instruction-domain representation. If M3 wins, the main story is that a small supervised model can learn this particular target-model boundary directly. In this run, M4 is clearly useful but does not beat M3.

The strict protocol also retrains the raw-prompt tokenizer only on the atomic train prompts. This avoids the caveat that a tokenizer trained under a different split may have seen text from the atomic held-out test prompts.

## A Strong Baseline

Before the neural models, M1 already does meaningful boundary prediction. Across split types, it beats the positive-rate baseline by a large margin.

![Figure 3: M1 split AUPRC](assets/figure3_m1_split_auprc.png)

*What to look for: The comparison to the positive-rate baseline is the key reference point; M1 is already far above random ranking across splits.*

This matters because TF-IDF is cheap, stable, and hard to beat. Any larger boundary model has to justify itself against this baseline.

## Clean Main Result

The main result uses only the strict atomic-train tokenizer.

![Figure 4: Strict atomic-tokenizer main result](assets/figure4_strict_main_result.png)

*What to look for: The main comparison is whether any small prompt-only model beats the 14.8% baseline under strict atomic OOD, not just which neural config is highest.*

The key test AUPRC numbers are:

| Model/config | Test AUPRC |
|---|---:|
| Positive-rate baseline | 14.8% |
| M1 TF-IDF full | 36.6% |
| M3 mean 40k | 43.7% ± 3.2% |
| M3 mean full | 46.1% ± 7.8% |
| M4 frozen 40k | 41.7% ± 5.2% |
| M4 frozen full | 39.8% ± 2.3% |

M3 mean full is the best average result in this strict protocol, but it is also seed-sensitive. M3 mean 40k is slightly lower and more stable. M4 improves over M1, but it does not beat M3.

I would read the M3 full number as the strongest evidence that the boundary is compressible, and the M3 40k number as the more conservative neural reference point. The two agree on the main conclusion: a tiny prompt-only encoder can beat the TF-IDF baseline under strict atomic OOD, but the exact neural ranking is still noisy.

The takeaway is not "bigger is automatically better." The frozen IF-domain encoder is useful, but in this setup the simpler supervised M3 encoder is stronger. One plausible explanation is that the frozen encoder gives a generally good instruction representation, while M3 can adapt all of its parameters to the exact checker boundary and target model used here.

## Tokenizer Protocol Audit

Earlier experiments used an old group-key tokenizer. That tokenizer was unsupervised, but it was trained under a different split protocol. For a clean atomic held-out story, I retrained the tokenizer only on atomic train prompts and reran the key models.

![Figure 5: Tokenizer protocol audit](assets/figure5_tokenizer_protocol_audit.png)

*What to look for: The stricter tokenizer removes a possible preprocessing caveat without weakening the main conclusion.*

The strict tokenizer did not weaken the result. In several cases it improved it:

| Config | Old group-key tokenizer AUPRC | Strict atomic-train tokenizer AUPRC |
|---|---:|---:|
| M1 full | 37.0% | 36.6% |
| M3 40k | 45.1% | 43.7% |
| M3 full | 35.0% | 46.1% |
| M4 20k | 35.3% | 39.5% |
| M4 40k | 36.9% | 41.7% |
| M4 full | 38.7% | 39.8% |

The tokenizer diagnostic does not suggest a length or token-support artifact. Both tokenizers have almost identical prompt lengths, p95 length is 577 tokens, and truncation is about 0.24%.

| Tokenizer protocol | Vocab size | Fit prompt count | Fit split | Rows | Mean token length | p95 token length | Max token length | Truncation rate | Unseen test token IDs | Unseen token mass |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| old group-key | 8,000 | - | group-key train | 82,806 | 227.9 | 577 | 5,786 | 0.243% | 3 | 1.78e-6 |
| strict atomic-train | 8,000 | 61,655 | atomic train | 82,806 | 228.0 | 577 | 5,527 | 0.242% | 3 | 1.77e-6 |

*What to look for: Prompt lengths and truncation rates are nearly unchanged, so the tokenizer result is unlikely to be a length-support artifact.*

So the clean strict protocol is not merely a stricter but weaker variant. It is the right protocol for the main claim.

## Validation Selection vs Test Oracle

Because this study involved iterative development, I wanted a simple anti-cherry-picking check: what would validation select, and what is the test oracle best result?

| Model family | Selection rule | Selected config | Val AUPRC | Test AUROC | Test AUPRC | Test Brier | Test ECE |
|---|---|---|---:|---:|---:|---:|---:|
| M1 TF-IDF | validation-selected by val AUPRC | baseline full | 80.0% | 83.1% | 36.6% | 16.4% | 20.5% |
| M1 TF-IDF | oracle-best by test AUPRC | baseline full | 80.0% | 83.1% | 36.6% | 16.4% | 20.5% |
| M3 mean | validation-selected by val AUPRC | mean pooling full | 80.7% | 84.4% | 46.1% | 16.0% | 14.5% |
| M3 mean | oracle-best by test AUPRC | mean pooling full | 80.7% | 84.4% | 46.1% | 16.0% | 14.5% |
| M4 frozen | validation-selected by val AUPRC | frozen encoder full | 79.2% | 82.5% | 39.8% | 16.7% | 16.5% |
| M4 frozen | oracle-best by test AUPRC | frozen encoder 40k | 77.6% | 82.2% | 41.7% | 16.0% | 15.2% |

*What to look for: Validation selection and test-oracle selection mostly agree for M3, while M4 shows why oracle-best numbers should be treated as diagnostics.*

For M1 there is only one strict config. For M3, validation selection and test oracle both select full. For M4, validation selects full, while the test oracle selects 40k. That is why I treat M4 40k as an interesting diagnostic point, but not as a clean validation-selected winner.

The calibration metrics are also useful here. M3 full has lower ECE than M1, but none of these models are well calibrated out of the box.

## Strict Key-Point Learning Curve

The strict tokenizer run is not a full learning curve. It has only the key points needed for the final comparison. Still, the sparse trend is informative.

![Figure 7: Strict tokenizer key curve](assets/figure7_strict_tokenizer_key_curve.png)

*What to look for: This is a sparse key-point curve, not a full scaling law; it is mainly a check that the strict tokenizer result is not a one-off point.*

M3 benefits from moving from 40k to full on average. M4 improves from 20k to 40k, but full does not improve test AUPRC. This matches the broader development pattern: validation performance can keep improving while atomic held-out test ranking does not.

## Selective Prediction

A boundary model is useful even if it is imperfect at 100% coverage. One natural use is high-confidence filtering: only act on prompts where the model is most confident.

![Figure 8: Strict selective accuracy](assets/figure8_strict_selective_accuracy.png)

*What to look for: Selective accuracy shows that the score is useful for filtering, but it can be inflated by the dominant negative class.*

At 50% coverage, M3 full reaches about 94.5% selective accuracy, compared with about 78.6% accuracy at full coverage. M4 40k reaches about 92.2% at 50% coverage.

Selective accuracy can be dominated by the negative class, so I also measured precision among the top predicted-pass prompts.

![Figure 9: Strict top-k precision](assets/figure9_strict_topk_precision.png)

*What to look for: Top-k precision is the more direct metric when the operational goal is to find prompts likely to pass.*

Top-k precision is the more important operational metric if the goal is to find prompts likely to pass. At top-100, M3 full averages 66.3% precision, M3 40k averages 55.7%, M4 40k averages 56.3%, and M1 is 25.0%. The variance is high at small k, but the neural models clearly rank pass-likely prompts better than M1.

## Calibration

The models are ranking useful signals, but their probabilities are not calibrated.

![Figure 10: Calibration](assets/figure10_strict_calibration.png)

*What to look for: The scores are useful for ordering prompts, but the reliability curves warn against treating them as literal probabilities.*

All models are over-confident in high predicted-probability regions. The reliability curves sit below the diagonal for most bins. This means the scores are useful for ranking and selection, but should not be interpreted as calibrated probabilities without post-hoc calibration.

The ECE values on test are:

| Model/config | Test ECE |
|---|---:|
| M1 full | 20.5% |
| M3 full | 14.5% |
| M4 40k | 15.2% |
| M4 full | 16.5% |

M3 full is best among these, but 14.5% ECE is still large.

## What M1 Learns

The M1 model is simple enough to inspect. Its strongest token n-grams map to ordinary prompt fragments such as punctuation, "appear at least", "keyword", "word", "in your response", and "each sentence".

![Figure 11: M1 feature coefficients](assets/figure11_m1_feature_coefficients.png)

*What to look for: The strongest features are ordinary prompt fragments, which supports the interpretation that many failures have surface-level correlates.*

This is a useful sanity check. The baseline is not magic. It is picking up regularities in constraint phrasing and format requirements. That also explains why it is strong: many checker failures are tied to explicit surface constraints in the prompt.

## Which Constraints Are Hard?

The atomic held-out test is not uniform. Some constraints are much easier to rank than others.

![Figure 12: Per-constraint AUPRC](assets/figure12_per_constraint_auprc.png)

*What to look for: The boundary is compressible on average, but the difficulty varies sharply by constraint family.*

`detectable_format:title` is relatively easy: all models perform strongly, and M3 is best. Some constraints such as `copy:copying_multiple` and `length_constraints:nth_paragraph_first_word` are much harder. For several keyword and punctuation constraints, M3 or M4 improves over M1, but the best model differs by constraint.

This supports a more cautious interpretation: the boundary is compressible, but not uniformly. The hard cases are real.

## Error Analysis

The v0.2 artifact includes a small prompt-level error table in `results/tables/prompt_level_error_examples.csv`. The CSV and README now include the evaluated input prompt and target-model response for these 12 selected examples; the compact table below keeps only metadata so the blog remains readable.

The rows below are sampled from strict atomic held-out test predictions for M3 mean full, with probabilities averaged across available seeds.

| Case | Group | Outcome | Prompt ID | y / pred | p(pass) | Constraint focus |
|---|---|---|---|---:|---:|---|
| E01 | High-confidence TP | TP | `26dea49c30ad` | 1 / 1 | 0.9678 | `detectable_format:title` |
| E02 | High-confidence TP | TP | `5761bb80f022` | 1 / 1 | 0.9678 | `detectable_format:title`<br>`keywords:existence` |
| E03 | High-confidence FP | FP | `d9018ce1fea1` | 0 / 1 | 0.9685 | `detectable_format:title` |
| E04 | High-confidence FP | FP | `a1ce25f3bf94` | 0 / 1 | 0.9678 | `keywords:no_adjacent_consecutive` |
| E05 | High-confidence FN | FN | `2366118e2d10` | 1 / 0 | 0.0056 | `keywords:no_adjacent_consecutive`<br>`startend:end_checker`<br>`first_word:first_word_answer`<br>`count:count_unique` |
| E06 | High-confidence FN | FN | `a30db5774417` | 1 / 0 | 0.0066 | `keywords:word_once`<br>`detectable_format:square_brackets` |
| E07 | High-confidence TN | TN | `1a2e9298819a` | 0 / 0 | 0.0024 | `copy:copying_simple`<br>`count:counting_composition`<br>`count:lowercase_counting`<br>`keywords:no_adjacent_consecutive` |
| E08 | High-confidence TN | TN | `c75e06ed4605` | 0 / 0 | 0.0024 | `copy:copying_multiple`<br>`count:counting_composition` |
| E09 | Hard constraint | FN | `419aff8c6a8d` | 1 / 0 | 0.0375 | `length_constraints:nth_paragraph_first_word` |
| E10 | Hard constraint | FN | `d6f46faf5733` | 1 / 0 | 0.0290 | `detectable_format:sentence_hyphens` |
| E11 | Hard constraint | FN | `67373ef5b455` | 1 / 0 | 0.0074 | `copy:copying_multiple` |
| E12 | Hard constraint | FN | `795d30a9ea1e` | 1 / 0 | 0.0463 | `first_word:first_word_sent` |

These examples are not a substitute for reading the original prompts locally, but they make the aggregate story easier to inspect: the model is confident in some surface-format regions, can be overconfident on familiar-looking constraints, and can under-rank rare or structured constraints even when the target model passes.

| Mode | Evidence in this run | Interpretation |
|---|---|---|
| Easy surface-format regions | M1 feature coefficients and strong `detectable_format:title` AUPRC | Some checker outcomes are strongly tied to visible prompt fragments such as title, keyword, punctuation, and sentence-form constraints. |
| False-positive risk | Keyword and punctuation fragments can look easy to a prompt-only model | A prompt may contain familiar pass-associated wording, while the target model still misses an exact count, required token, or formatting detail. |
| False-negative risk | Rare held-out atomic constraints have little direct training support | The boundary model may down-rank unfamiliar constraint phrasings even when the target model happens to comply. |
| Genuine hard cases | `copy:copying_multiple` and `length_constraints:nth_paragraph_first_word` remain difficult | These tasks require more structured response behavior than simple surface cues can reliably predict. |

## Exploratory Results

The old-tokenizer experiments are useful for understanding the research path, but I would not use them as the clean benchmark.

![Appendix A1: Old tokenizer learning curves](assets/appendix_a1_old_tokenizer_learning_curves.png)

*What to look for: These older runs are useful for understanding the research path, but they are not the clean benchmark.*

Under the old tokenizer, M3 peaked around 40k and then dropped at full on atomic held-out test. That motivated the seed-sensitivity and validation/test mismatch analysis.

I also tried CLS pooling and larger M3 capacity variants under the old tokenizer protocol.

![Appendix A2: M3 capacity and pooling](assets/appendix_a2_m3_capacity_pooling.png)

*What to look for: The larger or alternative M3 variants did not give stable enough gains to change the model-family story.*

The capacity variants did not produce stable enough gains to justify further scaling.

The fixed-sample 55k experiment showed substantial training-seed sensitivity.

![Appendix A4: M3 55k seed stability](assets/appendix_a4_m3_55k_seed_stability.png)

*What to look for: The seed spread is a warning that neural-model ranking should be interpreted with variance, not as a single deterministic number.*

This is one reason I stopped expanding the model family and focused on protocol cleanup. At this point, the main uncertainty was no longer whether a larger model family could be tried, but whether the protocol was clean enough and whether the observed boundary signal was robust under stricter preprocessing and selection rules.

## Dataset and IFEval attribution

This study uses IFEval-style verifiable instruction-following tasks. IFEval was introduced by Zhou et al. as an objective and reproducible benchmark for instruction following, based on automatically checkable constraints such as length, keyword, format, punctuation, and similar rules.

This post does not report a standard IFEval benchmark score. Instead, it uses the IFEval-style checker setting to define a different supervised problem: given only the prompt `x`, predict whether a fixed local target model `T` will produce an answer that passes the deterministic checker.

Where applicable, the prompt format, constraint families, and checker logic are derived from or inspired by the original IFEval code and data. Any modifications in this experiment include target-model relabeling with Qwen3-4B-Instruct-2507, split construction for boundary prediction, and supervised training of prompt-only boundary models.

## References

- Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., & Hou, L. (2023). Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911.
- Google Research IFEval implementation and data: `google-research/google-research/instruction_following_eval`.

The original IFEval data is released under CC BY 4.0, and the source code is released under Apache 2.0 by Google Research. This work uses it only as the instruction/checker source for constructing target-specific boundary labels.

## Limitations

This is an exploratory study, not a leaderboard.

Important limitations:

- It uses one target model.
- It uses one task family: verifiable instruction-following prompts.
- The atomic held-out test set was inspected during model development.
- The labels depend on a deterministic checker, not human preference.
- The boundary model predicts checker pass/fail, not whether the target response is generally useful.
- High AUPRC does not mean the model "understands" the instruction. It may learn prompt-surface regularities that correlate with target-model failures.
- The probabilities are not calibrated enough to use as literal probabilities without calibration.

The third limitation is the most important protocol caveat. Because the atomic held-out test set influenced development decisions, I treat the result as evidence for compressibility rather than as a final benchmark number. A stronger follow-up would freeze the protocol in advance, add a new untouched held-out split, and repeat the evaluation on at least one additional target model.

## Takeaways

1. Low-coupling LLM behavior boundaries can be highly compressible.
2. TF-IDF logistic regression is a surprisingly strong baseline.
3. Tiny supervised encoders improve AUPRC under strict atomic OOD.
4. Frozen IF-domain pretraining is not automatically better.
5. Split-specific preprocessing and tokenization matter.
6. Ranking is more reliable than calibration in the current models.

The most defensible headline result is:

```text
Positive-rate baseline AUPRC: 14.8%
M1 strict TF-IDF full AUPRC: 36.6%
M3 strict mean full AUPRC: 46.1% ± 7.8%
M3 strict mean 40k AUPRC: 43.7% ± 3.2%
M4 strict frozen 40k AUPRC: 41.7% ± 5.2%
```

That is enough to support the core claim: for this class of verifiable instructions, the target model's pass/fail boundary is much smaller than the target model itself.

## Reproducibility Notes

The final blog assets are in:

```text
boundary-if/runs/blog_final_assets/
```

Core files:

```text
run_metrics.csv
run_metrics_by_config.csv
selection_table.csv
predictions.parquet
selective_metrics.csv
topk_metrics.csv
calibration_bins.csv
per_constraint_metrics.csv
tokenizer_audit.csv
feature_coefficients.csv
```

The generation script is:

```text
boundary-if/scripts/generate_blog_final_assets.py
```

For a release intended to be reproduced outside my environment, I would also pin the target decoding configuration, maximum output length, checker version or commit hash, random seeds, optimizer settings, learning rate, batch size, epoch schedule, M3/M4 architecture details, and the exact command lines used to generate labels and train each boundary model.

This Markdown embeds the figures as base64 images for portability. That is convenient for a single-file draft, but for publication I would export the figures as separate PNG or WebP files and reference them by path or CDN URL. That keeps the post smaller and makes diffs easier to review.
