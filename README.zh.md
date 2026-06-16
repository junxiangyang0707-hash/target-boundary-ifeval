# IFEval 风格检查下的目标模型边界预测

这是一个可复现实验工程和技术博客材料库，用于整理一个小型边界模型研究。

核心问题是：

> 在真正运行固定本地 LLM 之前，一个只看 prompt 的小模型能否预测该目标模型是否会通过确定性的 IFEval 风格指令检查？

本实验中的答案是：可以，但需要明确 caveat。在严格的 atomic held-out split 上，正例率 AUPRC 基线是 14.8%，M1 TF-IDF 达到 36.6%，一个很小的监督式双向 Transformer 达到 46.1% ± 7.8% AUPRC。

## 这个仓库是什么

这个仓库定位为：

- 技术博客材料库
- 博客图表的轻量复现 artifact
- split / tokenizer / 结果表的可审计记录
- 在具备本地数据和算力时可重新运行实验的代码与配置

它不是 IFEval 排行榜提交。它预测的是一个固定目标模型 Qwen3-4B-Instruct-2507 在确定性 checker 下的 pass/fail 边界。

## 主结果

| Model/config | Test AUPRC |
|---|---:|
| Positive-rate baseline | 14.8% |
| M1 TF-IDF full | 36.6% |
| M3 mean 40k | 43.7% ± 3.2% |
| M3 mean full | 46.1% ± 7.8% |
| M4 frozen 40k | 41.7% ± 5.2% |
| M4 frozen full | 39.8% ± 2.3% |

![Strict atomic-tokenizer multiseed test results](docs/assets/figure4_strict_main_result.png)

## 仓库结构

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

## 版本范围

v0.1 包含：

- 英文和中文博客草稿。
- 博客图片。
- attribution 和 citation 元数据。

v0.2 包含：

- v0.1 全部内容。
- 聚合指标表。
- 博客分析用到的 validation/test predictions。
- split manifests。
- 代码、配置、测试和 Docker 文件。

默认不包含：

- raw promptsets
- target model 原始输出
- 本地模型 checkpoint
- 目标模型权重
- Hugging Face / vLLM cache
- W&B run 目录
- 本地 chat logs 和开发过程记录

## 包含的结果文件

核心表位于 `results/tables/`：

- `run_metrics.csv`
- `run_metrics_by_config.csv`
- `selection_table.csv`
- `split_summary.csv`
- `tokenizer_audit.csv`
- `selective_metrics.csv`
- `topk_metrics.csv`
- `per_constraint_metrics.csv`
- `feature_coefficients.csv`

最终博客分析使用的 predictions 位于：

```text
results/predictions/strict_atomic_blog_predictions.parquet
```

split manifests 位于：

```text
data/splits/
```

这些 artifact 不包含 prompt 文本，也不包含目标模型原始输出。

## 运行代码

原始 workflow 以 Docker 为主：

```bash
cd boundary-if
docker compose build app
docker compose run --rm app pytest
```

GPU 实验：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm app pytest
```

完整 target-model relabeling 需要本地模型权重和 vLLM 兼容 GPU 环境。当前发布的轻量 artifact 已足够检查报告中的 metrics 和 figures，不需要重新跑目标模型推理。

## 博客

- English: [`docs/boundary_prediction_en.md`](docs/boundary_prediction_en.md)
- 中文: [`docs/boundary_prediction_zh.md`](docs/boundary_prediction_zh.md)

## Attribution

见 [`ATTRIBUTION.md`](ATTRIBUTION.md)。本项目使用 IFEval 风格任务和 checker logic 作为 instruction/checker source，用于构造特定目标模型的边界标签。本文不报告标准 IFEval benchmark 分数。

## License

除特别说明外，代码以 Apache-2.0 发布。博客文本、图片和表格型研究 artifact 可在署名条件下按 CC BY 4.0 复用。Vendored IFEval checker 文件保留原始 Google Research Apache-2.0 headers。
