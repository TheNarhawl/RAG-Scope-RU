# Experiment results

_Сгенерировано: 2026-10-28 00:40:29_

Всего прогонов: **7**

| # | Tag | Embedding | Chunk | Overlap | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | ms/q | Timestamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qdrant_minilm | all-MiniLM-L6-v2 | 512 | 64 | 0.130 | 0.225 | 0.305 | — | 0.188 | — | 20261017_000037 |
| 2 | e5_check | multilingual-e5-large | 512 | 64 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | 48.9 | 20261017_020038 |
| 3 | e5-dense | multilingual-e5-large | 512 | 64 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | 46.4 | 20261017_065800 |
| 4 | e5-hybrid-rerank | multilingual-e5-large | 512 | 64 | 0.655 | 0.840 | 0.895 | 0.935 | 0.755 | 290.4 | 20261017_070054 |
| 5 | e5-hybrid | multilingual-e5-large | 512 | 64 | 0.805 | 0.900 | 0.920 | 0.945 | 0.854 | 50.0 | 20261017_071007 |
| 6 | ab-e5-naive | multilingual-e5-large | 512 | 64 | 0.800 | 0.895 | 0.920 | 0.945 | 0.852 | 54.3 | 20261027_082441 |
| 7 | ab-e5-structural | multilingual-e5-large | 512 | 64 | 0.825 | 0.905 | 0.935 | 0.960 | 0.870 | 56.1 | 20261027_091020 |

## Best run (по MRR)

- **Tag:** `ab-e5-structural`
- **Embedding:** `intfloat/multilingual-e5-large`
- **Chunk:** 512 / overlap 64
- **MRR:** 0.870
- **Hit@1:** 0.825
- **Файл:** `metrics_ab-e5-structural_20261027_091020.json`
