# Experiment results

_Сгенерировано: 2026-10-17 01:46:03_

Всего прогонов: **2**

| # | Tag | Embedding | Chunk | Overlap | N | Hit@1 | Hit@3 | Hit@5 | MRR | Timestamp |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qdrant_minilm | all-MiniLM-L6-v2 | 512 | 64 | 200 | 0.130 | 0.225 | 0.305 | 0.188 | 20261017_000037 |
| 2 | qdrant_e5_large | multilingual-e5-large | 512 | 64 | 200 | 0.730 | 0.860 | 0.895 | 0.794 | 20261017_014537 |

## Best run (по MRR)

- **Tag:** `qdrant_e5_large`
- **Embedding:** `intfloat/multilingual-e5-large`
- **Chunk:** 512 / overlap 64
- **MRR:** 0.794
- **Hit@1:** 0.730
- **Файл:** `metrics_qdrant_e5_large_20261017_014537.json`
