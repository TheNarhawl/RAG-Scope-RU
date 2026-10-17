# Experiment results

_Сгенерировано: 2026-10-17 10:04:48_

Всего прогонов: **5**

| # | Tag | Embedding | Chunk | Overlap | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Faithful | AnsCorr | ms/q | Timestamp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qdrant_minilm | all-MiniLM-L6-v2 | 512 | 64 | 200 | 0.130 | 0.225 | 0.305 | — | 0.188 | — | — | — | 20261017_000037 |
| 2 | e5_check | multilingual-e5-large | 512 | 64 | 200 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | — | — | 48.9 | 20261017_020038 |
| 3 | e5-dense | multilingual-e5-large | 512 | 64 | 200 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | — | — | 46.4 | 20261017_065800 |
| 4 | e5-hybrid-rerank | multilingual-e5-large | 512 | 64 | 200 | 0.655 | 0.840 | 0.895 | 0.935 | 0.755 | — | — | 290.4 | 20261017_070054 |
| 5 | e5-hybrid | multilingual-e5-large | 512 | 64 | 200 | 0.775 | 0.860 | 0.890 | 0.925 | 0.825 | — | — | 48.9 | 20261017_070352 |

Faithful и AnsCorr (RAGAS) появятся после подключения оценки генерации через LLM + RAGAS.


## Best run (по MRR)

- **Tag:** `e5_check`
- **Embedding:** `intfloat/multilingual-e5-large`
- **Chunk:** 512 / overlap 64
- **MRR:** 0.825
- **Hit@1:** 0.775
- **Файл:** `metrics_e5_check_20261017_020038.json`
