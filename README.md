# RAG-Scope-RU

<p align="left">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x%20CUDA-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/Vector_DB-Qdrant-DC2626.svg?logo=qdrant&logoColor=white" alt="Qdrant"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-FFD21E.svg" alt="Hugging Face"></a>
  <a href="https://docs.ragas.io/"><img src="https://img.shields.io/badge/Eval-Ragas_LLM--as--a--Judge-8B5CF6.svg" alt="Ragas"></a>
  <a href="https://github.com/TheNarhawl/RAG-Scope-RU/actions"><img src="https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=githubactions&logoColor=white" alt="GitHub Actions"></a>
  <a href="docker/"><img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white" alt="Docker"></a>
</p>

Модульная экспериментальная платформа (песочница) для исследования и оценки качества этапа поиска (Retrieval) и сквозной генерации в RAG-системах на русскоязычных нормативно-технических документах.

Платформа позволяет изолированно менять и замерять влияние ключевых параметров пайплайна:
* **Модели эмбеддингов**: от легковесных до SOTA мультиязычных и адаптированных под русский язык
* **Стратегии поиска**: плотный (Dense), разреженный лексический (BM25) и гибридный с Reciprocal Rank Fusion (RRF / FusionQuery)
* **Переранжирование кандидатов**: Cross-Encoder Reranker (`bge-reranker-v2-m3`, `bge-reranker-base`)
* **Парсинг документов**: A/B сравнение стандартного PyPDF и Markdown-структурированного PyMuPDF4LLM
* **Параметры нарезки текста**: размер чанка и перекрытие, включая структурный чанкинг по заголовкам и таблицам
* **Метрики качества извлечения**: Hit Rate@K, MRR и задержка ответа (ms/query)
* **Сквозная оценка качества ответов**: Триада метрик RAGAS (Faithfulness, Answer Correctness, Context Recall) с LLM-as-a-Judge

---

## 🛠 Технологический стек

| Область | Технологии / Инструменты | Назначение в проекте |
| :--- | :--- | :--- |
| **Векторная база данных** | **[Qdrant](https://qdrant.tech/)** `1.12+` (Docker / Local / Remote) | Индексация dense и sparse векторов, гибридный поиск через Prefetch + RRF (`FusionQuery`), изоляция коллекций |
| **Нейросетевые эмбеддинги** | **[Sentence-Transformers](https://sbert.net/)**, **[PyTorch](https://pytorch.org/)** (CUDA 12) | Инференс плотных векторных представлений текста (до 1024 dim) с автовыбором GPU/CPU |
| **Модели представления (Dense)** | `deepvk/USER-bge-m3`, `BAAI/bge-m3`, `intfloat/multilingual-e5-*`, `Alibaba-NLP/gte-multilingual-base`, `cointegrated/rubert-tiny2` | Сравнительный анализ точности семантического поиска на русском нормативном корпусе |
| **Лексический / Разреженный поиск** | **[FastEmbed](https://github.com/qdrant/fastembed)**, `rank-bm25` | Генерация sparse-индексов BM25 / SPLADE с модификатором IDF в Qdrant |
| **Переранжирование (Rerank)** | **Cross-Encoder** (`BAAI/bge-reranker-v2-m3`, `bge-reranker-base`) | Вторичный точный скоринг топ-кандидатов с оценкой семантической близости пары «запрос-пассаж» |
| **Парсинг и извлечение данных** | **[PyMuPDF4LLM](https://github.com/pymupdf/PyMuPDF)** (fitz), **[PyPDF](https://pypdf.readthedocs.io/)** | Извлечение текста нормативных PDF с сохранением Markdown-заголовков, списков и таблиц |
| **Чанкинг и токенизация** | Собственные сплиттеры: `structural_chunk_text` и `naive_chunk_text` | Разделение по границам предложений и Markdown-блоков (`#`, `##`, `|`) с контекстным перекрытием (overlap) |
| **Генерация и LLM-Judge (RAGAS)** | **[Ragas](https://docs.ragas.io/)**, **[OpenAI API](https://platform.openai.com/)** (совместимость с DeepSeek, Ollama, vLLM) | Двухэтапная валидация: скрининг поиска + генерация RAG-ответов и расчет триады метрик |
| **Метрики и бенчмаркинг** | **[RRNCB Benchmark](https://huggingface.co/datasets/FractalGPT/RRNCBPublic)**, `Hit@K`, `MRR@K`, `ms/query` | 200 контрольных экспертных вопросов по ГОСТам, СНиПам и законодательству РФ |
| **Инфраструктура и MLOps** | **Docker**, **Docker Compose**, **GitHub Actions**, **Google Colab** | Контейнеризация сервисов, автоматический CI/CD (flake8, матричные проверки, pytest), стенды на GPU Tesla T4 |
| **Аналитика и визуализация** | **Pandas**, **Matplotlib**, **Graphify** | Построение сводных Markdown-отчетов, Парето-фронтов («Качество vs Задержка») и графа связности кода |

---

## Быстрый старт

### 1. Окружение и зависимости

```bash
# Клонирование и переход в проект
git clone <URL_РЕПОЗИТОРИЯ>
cd RAG-Scope-RU

# Создание и активация виртуального окружения
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Заполните QDRANT_URL и HF_TOKEN (при необходимости)
```

### 2. Подготовка данных

Скрипт `src/pipelines/prepare_data.py` в автоматическом режиме:
1. Выкачивает CSV-датасет RRNCB с Hugging Face и конвертирует его в `data/eval/questions.json`.
2. Распаковывает корпус из 65 нормативных PDF-документов из архива `docs.zip` в `data/docs/`.

```bash
python src/pipelines/prepare_data.py
```

### 3. Генерация матрицы конфигураций

Генерирует полную сетку YAML-конфигов под разные модели, размеры чанков и стратегии поиска:

```bash
python make_configs.py
```

### 4. Индексация и запуск экспериментов

Индексация выполняется один раз на пару **«Модель + Размер чанка»**, после чего оцениваются все стратегии поиска. Скрипт `run_grid.py` поддерживает фильтрацию и автоматически пропускает уже посчитанные конфигурации:

```bash
# Запуск всей сетки или фильтрация по конкретной модели (например, gte_multilingual или labse_ru)
python src/pipelines/run_grid.py --filter gte_multilingual

# Индексация документов вручную (Dense + Sparse BM25 в Qdrant)
python src/pipelines/ingest.py --config e5_large/c512/dense

# Проверка поиска вручную
python src/pipelines/retrieve.py --config e5_large/c512/hybrid "срок исковой давности"

# Оценка метрик ранжирования (Stage 1: Hit@K, MRR, latency)
python src/pipelines/evaluate.py --config e5_large/c512/dense
python src/pipelines/evaluate.py --config e5_large/c512/hybrid
python src/pipelines/evaluate.py --config e5_large/c512/hybrid-rerank

# Оценка качества генерации (Stage 2: RAGAS Faithfulness, AnsCorr, Context Recall)
python src/pipelines/evaluate.py --config e5_large/c512/hybrid-rerank --with-ragas --ragas-limit 20

# Сборка единого отчета по всем запускам в experiments/RESULTS.md
python src/pipelines/summarize_experiments.py
```

### 5. Запуск A/B тестирования парсеров (pypdf vs pymupdf4llm)

Сравнение качества извлечения на трех весовых категориях моделей (`rubert_tiny2`, `user_bge_m3`, `bge_m3`, `e5_large`):

```bash
# Быстрый A/B скрининг поиска (Hit@K, MRR)
python src/pipelines/run_ab_experiment.py --group ab_parsing

# A/B прогон со сквозной оценкой RAGAS
python src/pipelines/run_ab_experiment.py --group ab_parsing --with-ragas --ragas-limit 20
```

### 6. Анализ A/B эксперимента (PyPDF vs PyMuPDF4LLM)

Для детального статистического анализа результатов парсинга подготовлен аналитический блокнот [`notebooks/ab_pypdf_vs_pymupdf.ipynb`](notebooks/ab_pypdf_vs_pymupdf.ipynb):
* Загрузка и сопоставление 8 экспериментов (4 пары моделей по 200 контрольных вопросов)
* Сводные карточки метрик с расчетом абсолютной дельты ($\Delta$) и относительного прироста (Lift %)
* Непараметрические статистические критерии: парный тест Уилкоксона для $RR$ и точный тест Мак-Немара для $Hit@1$, $Hit@5$
* Построение 95% бутстрап доверительных интервалов и Forest Plot
* Анализ структуры побед и поражений (Win / Tie / Loss) и качественный разбор выигранных/проигранных вопросов

---

## 🏗 Архитектурные особенности и надежность

* **Двухэтапная валидация (Two-Stage Evaluation):**
  * **Stage 1 (Retrieval Screening):** быстрый замер качества поиска (Hit@1..10, MRR, latency) по всей матрице конфигураций без затрат на инференс больших LLM.
  * **Stage 2 (Generation & RAGAS):** точечная оценка контекстной генерации (Faithfulness, Answer Correctness, Context Recall) только для конфигураций Парето-фронта.
* **Гибридный поиск с защитой от нулевых векторов:** совмещение Dense-поиска и разреженного BM25/SPLADE через `FusionQuery` (RRF) в Qdrant с fallback-логикой при редких или отсутствующих токенах.
* **Борьба с утечками ресурсов:** автоматическая очистка промежуточных коллекций в памяти Qdrant и сброс CUDA кэша (`torch.cuda.empty_cache()`) между экспериментами.
* **Совместимость с Transformers 5+:** встроенное динамическое восстановление неперсистентных буферов RoPE и `position_ids` (патч для `gte-multilingual-base` и кастомных архитектур).
* **Абсолютная воспроизводимость:** сквозная фиксация сидов (`seed=42`) в PyTorch, NumPy и системном рандомайзере.

---

## Датасет

Эксперименты проводятся на бенчмарке **RRNCB** (Russian RAG Normative Corporate Benchmark):
* **Корпус:** 65 нормативно-технических PDF (ГОСТы, СНиПы, федеральные законы, кодексы).
* **Тестовая выборка:** 200 контрольных вопросов с эталонными ответами и ссылкой на исходный документ.
* **Hugging Face:** [FractalGPT/RRNCBPublic](https://huggingface.co/datasets/FractalGPT/RRNCBPublic)

---

## Структура проекта

```text
RAG-Scope-RU/
├── configs/                    # Иерархические YAML-конфиги экспериментов
│   ├── baseline.yaml           # Контрольный англоязычный бейслайн
│   ├── e5_large/               # Модель: intfloat/multilingual-e5-large
│   │   ├── c512/               # Размер чанка: 512 символов
│   │   │   ├── dense.yaml
│   │   │   ├── hybrid.yaml
│   │   │   └── hybrid-rerank.yaml
│   │   └── ...
│   ├── bge_m3/                 # Модель: BAAI/bge-m3
│   ├── user_bge_m3/            # Модель: deepvk/USER-bge-m3
│   └── ...
├── data/
│   ├── docs/                   # Корпус PDF (загружается автоматически, не в git)
│   └── eval/                   # dataset_sample.csv и questions.json
├── docker/
│   ├── Dockerfile              # Образ Python + PyTorch CUDA
│   └── docker-compose.yml      # Сервисы App + Qdrant (GPU)
├── scripts/
│   ├── reset-qdrant.ps1        # Полный сброс хранилища Qdrant
│   ├── restart-qdrant.ps1      # Мягкий перезапуск контейнера Qdrant
│   └── quickstart.txt          # Шпаргалка команд запуска
├── experiments/                # Результаты прогонов в JSON и сводный отчет
│   └── RESULTS.md              # Актуальная таблица метрик всех экспериментов
├── notebooks/
│   └── results.ipynb           # Аналитический Jupyter-ноутбук (лидерборд, Парето-фронт)
├── src/
│   ├── __init__.py
│   ├── common/                 # 1. Инфраструктурный и служебный слой
│   │   ├── __init__.py
│   │   ├── constants.py        # Константы (KS, дефолтные модели, сиды)
│   │   ├── paths.py            # Централизованные пути и чтение .env
│   │   └── utils.py            # Воспроизводимость, выбор девайса, кэши
│   ├── core/                   # 2. Алгоритмическое ядро (чистая логика)
│   │   ├── __init__.py
│   │   ├── splitters.py        # Алгоритмы нарезки: naive и structural
│   │   └── metrics.py          # Расчет Hit@K, MRR и вспомогательные функции
│   └── pipelines/              # 3. Исполняемые сценарии (CLI / точки входа)
│       ├── __init__.py
│       ├── prepare_data.py     # Выкачка датасета с HF и PDF из архива
│       ├── ingest.py           # Чанкинг, расчет эмбеддингов, загрузка в Qdrant
│       ├── retrieve.py         # Инференс поиска (Dense / Hybrid / Rerank)
│       ├── evaluate.py         # Пакетная оценка качества поиска на вопросах
│       ├── run_grid.py         # Оркестратор прогона координатной сетки
│       └── summarize_experiments.py # Построение сводной Markdown-таблицы
├── tests/                      # Юнит-тесты компонентов (pytest)
├── make_configs.py             # Автогенератор дерева конфигураций
├── requirements.txt            # Зависимости проекта
└── README.md
```

---

## Запуск через Docker

Контейнеризация предназначена как для изолированного запуска локально, так и для бесшовного переноса стенда на арендованные GPU-хосты (RunPod, Vast.ai, Cloud.ru, Selectel).

### Локально / на GPU-сервере

```bash
cd docker

# 1. Поднять сервис Qdrant
docker compose up -d qdrant

# 2. Собрать контейнер приложения
docker compose build app

# 3. Выкачать данные корпуса и вопросов внутри контейнера
docker compose run --rm app python src/pipelines/prepare_data.py

# 4. Сгенерировать матрицу конфигов
docker compose run --rm app python make_configs.py

# 5. Проиндексировать выбранную связку
docker compose run --rm app python src/pipelines/ingest.py --config e5_large/c512/dense

# 6. Запустить расчет метрик
docker compose run --rm app python src/pipelines/evaluate.py --config e5_large/c512/hybrid

# 7. Сформировать сводку
docker compose run --rm app python src/pipelines/summarize_experiments.py

# 8. Остановка окружения
docker compose down
```

## Текущие результаты экспериментов

Сводная таблица формируется автоматически скриптом `src/pipelines/summarize_experiments.py` в файле `experiments/RESULTS.md`.

Сгенерировано: 2026-10-28 10:35:25

Всего прогонов: **78**

| # | Tag | Embedding | Chunk | Overlap | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | ms/q |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | labse_ru_c256_dense | LaBSE-en-ru | 256 | 32 | 0.540 | 0.670 | 0.705 | 0.770 | 0.617 | 47.1 |
| 2 | e5_small_c1024_dense | multilingual-e5-small | 1024 | 128 | 0.730 | 0.840 | 0.875 | 0.920 | 0.792 | 18.2 |
| 3 | e5_small_c1024_hybrid | multilingual-e5-small | 1024 | 128 | 0.815 | 0.905 | 0.925 | 0.950 | 0.865 | 18.4 |
| 4 | e5_small_c1024_hybrid_rerank | multilingual-e5-small | 1024 | 128 | 0.880 | 0.930 | 0.950 | 0.960 | 0.908 | 1181.5 |
| 5 | labse_ru_c256_hybrid | LaBSE-en-ru | 256 | 32 | 0.690 | 0.885 | 0.915 | 0.925 | 0.789 | 47.3 |
| 6 | labse_ru_c256_hybrid_rerank | LaBSE-en-ru | 256 | 32 | 0.850 | 0.900 | 0.915 | 0.925 | 0.878 | 441.5 |
| 7 | e5_small_c256_dense | multilingual-e5-small | 256 | 32 | 0.765 | 0.880 | 0.900 | 0.925 | 0.825 | 18.9 |
| 8 | e5_small_c256_hybrid | multilingual-e5-small | 256 | 32 | 0.820 | 0.920 | 0.955 | 0.965 | 0.875 | 19.2 |
| 9 | e5_small_c256_hybrid_rerank | multilingual-e5-small | 256 | 32 | 0.845 | 0.930 | 0.940 | 0.965 | 0.888 | 405.4 |
| 10 | labse_ru_c512_dense | LaBSE-en-ru | 512 | 64 | 0.470 | 0.660 | 0.715 | 0.770 | 0.575 | 19.1 |
| 11 | labse_ru_c512_hybrid | LaBSE-en-ru | 512 | 64 | 0.680 | 0.865 | 0.915 | 0.940 | 0.780 | 19.1 |
| 12 | e5_small_c512_dense | multilingual-e5-small | 512 | 64 | 0.750 | 0.865 | 0.890 | 0.940 | 0.814 | 19.1 |
| 13 | labse_ru_c512_hybrid_rerank | LaBSE-en-ru | 512 | 64 | 0.840 | 0.915 | 0.925 | 0.950 | 0.879 | 658.2 |
| 14 | e5_small_c512_hybrid | multilingual-e5-small | 512 | 64 | 0.830 | 0.915 | 0.935 | 0.955 | 0.876 | 19.2 |
| 15 | e5_small_c512_hybrid_rerank | multilingual-e5-small | 512 | 64 | 0.855 | 0.930 | 0.935 | 0.955 | 0.891 | 670.8 |
| 16 | e5_base_c1024_dense | multilingual-e5-base | 1024 | 128 | 0.720 | 0.820 | 0.850 | 0.890 | 0.774 | 17.6 |
| 17 | e5_base_c1024_hybrid | multilingual-e5-base | 1024 | 128 | 0.780 | 0.890 | 0.920 | 0.950 | 0.841 | 18.7 |
| 18 | mpnet_multilingual_c256_dense | paraphrase-multilingual-mpnet-base-v2 | 256 | 32 | 0.635 | 0.750 | 0.785 | 0.855 | 0.702 | 40.4 |
| 19 | mpnet_multilingual_c256_hybrid | paraphrase-multilingual-mpnet-base-v2 | 256 | 32 | 0.760 | 0.895 | 0.910 | 0.950 | 0.833 | 19.9 |
| 20 | e5_base_c1024_hybrid_rerank | multilingual-e5-base | 1024 | 128 | 0.855 | 0.930 | 0.940 | 0.950 | 0.891 | 1185.2 |
| 21 | mpnet_multilingual_c256_hybrid_rerank | paraphrase-multilingual-mpnet-base-v2 | 256 | 32 | 0.850 | 0.915 | 0.930 | 0.955 | 0.885 | 410.7 |
| 22 | mpnet_multilingual_c512_dense | paraphrase-multilingual-mpnet-base-v2 | 512 | 64 | 0.565 | 0.710 | 0.780 | 0.810 | 0.650 | 33.8 |
| 23 | mpnet_multilingual_c512_hybrid | paraphrase-multilingual-mpnet-base-v2 | 512 | 64 | 0.710 | 0.850 | 0.895 | 0.940 | 0.795 | 20.3 |
| 24 | mpnet_multilingual_c512_hybrid_rerank | paraphrase-multilingual-mpnet-base-v2 | 512 | 64 | 0.835 | 0.910 | 0.930 | 0.940 | 0.874 | 658.8 |
| 25 | e5_base_c256_dense | multilingual-e5-base | 256 | 32 | 0.710 | 0.825 | 0.880 | 0.925 | 0.782 | 36.5 |
| 26 | e5_base_c256_hybrid | multilingual-e5-base | 256 | 32 | 0.785 | 0.915 | 0.955 | 0.970 | 0.856 | 35.2 |
| 27 | e5_base_c256_hybrid_rerank | multilingual-e5-base | 256 | 32 | 0.840 | 0.920 | 0.935 | 0.970 | 0.883 | 421.6 |
| 28 | e5_base_c512_dense | multilingual-e5-base | 512 | 64 | 0.720 | 0.810 | 0.860 | 0.915 | 0.775 | 35.4 |
| 29 | e5_base_c512_hybrid | multilingual-e5-base | 512 | 64 | 0.775 | 0.895 | 0.925 | 0.950 | 0.840 | 18.6 |
| 30 | e5_base_c512_hybrid_rerank | multilingual-e5-base | 512 | 64 | 0.835 | 0.920 | 0.940 | 0.950 | 0.879 | 671.7 |
| 31 | sbert_large_c1024_dense | sbert_large_nlu_ru | 1024 | 128 | 0.440 | 0.590 | 0.655 | 0.725 | 0.529 | 28.9 |
| 32 | sbert_large_c1024_hybrid | sbert_large_nlu_ru | 1024 | 128 | 0.635 | 0.845 | 0.865 | 0.940 | 0.752 | 29.0 |
| 33 | sbert_large_c1024_hybrid_rerank | sbert_large_nlu_ru | 1024 | 128 | 0.865 | 0.920 | 0.930 | 0.950 | 0.894 | 1138.8 |
| 34 | e5_large_c1024_dense | multilingual-e5-large | 1024 | 128 | 0.745 | 0.835 | 0.885 | 0.940 | 0.802 | 31.4 |
| 35 | e5_large_c1024_hybrid | multilingual-e5-large | 1024 | 128 | 0.820 | 0.895 | 0.915 | 0.945 | 0.862 | 30.8 |
| 36 | e5_large_c1024_hybrid_rerank | multilingual-e5-large | 1024 | 128 | 0.865 | 0.925 | 0.945 | 0.970 | 0.901 | 1209.8 |
| 37 | sbert_large_c256_dense | sbert_large_nlu_ru | 256 | 32 | 0.460 | 0.660 | 0.740 | 0.795 | 0.572 | 28.1 |
| 38 | sbert_large_c256_hybrid | sbert_large_nlu_ru | 256 | 32 | 0.645 | 0.885 | 0.915 | 0.950 | 0.771 | 29.2 |
| 39 | sbert_large_c256_hybrid_rerank | sbert_large_nlu_ru | 256 | 32 | 0.850 | 0.920 | 0.930 | 0.945 | 0.885 | 419.6 |
| 40 | rubert_tiny2_c1024_dense | rubert-tiny2 | 1024 | 128 | 0.380 | 0.590 | 0.660 | 0.715 | 0.495 | 9.5 |
| 41 | rubert_tiny2_c1024_hybrid | rubert-tiny2 | 1024 | 128 | 0.650 | 0.865 | 0.880 | 0.930 | 0.761 | 10.6 |
| 42 | rubert_tiny2_c1024_hybrid_rerank | rubert-tiny2 | 1024 | 128 | 0.855 | 0.915 | 0.940 | 0.960 | 0.890 | 1099.0 |
| 43 | rubert_tiny2_c256_dense | rubert-tiny2 | 256 | 32 | 0.465 | 0.620 | 0.700 | 0.790 | 0.567 | 20.1 |
| 44 | rubert_tiny2_c256_hybrid | rubert-tiny2 | 256 | 32 | 0.660 | 0.895 | 0.930 | 0.945 | 0.780 | 11.7 |
| 45 | e5_large_c256_dense | multilingual-e5-large | 256 | 32 | 0.795 | 0.885 | 0.905 | 0.935 | 0.843 | 30.4 |
| 46 | e5_large_c256_hybrid | multilingual-e5-large | 256 | 32 | 0.845 | 0.915 | 0.955 | 0.970 | 0.889 | 30.5 |
| 47 | rubert_tiny2_c256_hybrid_rerank | rubert-tiny2 | 256 | 32 | 0.845 | 0.910 | 0.940 | 0.955 | 0.883 | 392.3 |
| 48 | e5_large_c256_hybrid_rerank | multilingual-e5-large | 256 | 32 | 0.865 | 0.935 | 0.955 | 0.980 | 0.903 | 438.2 |
| 49 | sbert_large_c512_dense | sbert_large_nlu_ru | 512 | 64 | 0.475 | 0.645 | 0.710 | 0.765 | 0.571 | 51.6 |
| 50 | sbert_large_c512_hybrid | sbert_large_nlu_ru | 512 | 64 | 0.735 | 0.895 | 0.910 | 0.945 | 0.815 | 26.4 |
| 51 | sbert_large_c512_hybrid_rerank | sbert_large_nlu_ru | 512 | 64 | 0.835 | 0.925 | 0.930 | 0.945 | 0.878 | 661.8 |
| 52 | rubert_tiny2_c512_dense | rubert-tiny2 | 512 | 64 | 0.435 | 0.635 | 0.680 | 0.750 | 0.544 | 11.0 |
| 53 | rubert_tiny2_c512_hybrid | rubert-tiny2 | 512 | 64 | 0.670 | 0.860 | 0.900 | 0.940 | 0.775 | 11.6 |
| 54 | rubert_tiny2_c512_hybrid_rerank | rubert-tiny2 | 512 | 64 | 0.845 | 0.930 | 0.935 | 0.950 | 0.886 | 627.3 |
| 55 | e5_large_c512_dense | multilingual-e5-large | 512 | 64 | 0.740 | 0.870 | 0.920 | 0.930 | 0.812 | 54.4 |
| 56 | e5_large_c512_hybrid | multilingual-e5-large | 512 | 64 | 0.810 | 0.890 | 0.925 | 0.950 | 0.860 | 28.2 |
| 57 | e5_large_c512_hybrid_rerank | multilingual-e5-large | 512 | 64 | 0.835 | 0.920 | 0.930 | 0.945 | 0.877 | 686.3 |
| 58 | bge_m3_c1024_dense | bge-m3 | 1024 | 128 | 0.750 | 0.865 | 0.895 | 0.930 | 0.815 | 29.7 |
| 59 | bge_m3_c1024_hybrid | bge-m3 | 1024 | 128 | 0.830 | 0.900 | 0.935 | 0.960 | 0.875 | 29.3 |
| 60 | bge_m3_c1024_hybrid_rerank | bge-m3 | 1024 | 128 | 0.855 | 0.930 | 0.950 | 0.960 | 0.895 | 1168.9 |
| 61 | bge_m3_c2048_dense | bge-m3 | 2048 | 256 | 0.730 | 0.835 | 0.880 | 0.915 | 0.791 | 27.3 |
| 62 | bge_m3_c2048_hybrid | bge-m3 | 2048 | 256 | 0.785 | 0.875 | 0.920 | 0.950 | 0.840 | 27.2 |
| 63 | bge_m3_c2048_hybrid_rerank | bge-m3 | 2048 | 256 | 0.880 | 0.930 | 0.955 | 0.960 | 0.909 | 2260.3 |
| 64 | bge_m3_c256_dense | bge-m3 | 256 | 32 | 0.720 | 0.850 | 0.885 | 0.920 | 0.789 | 31.2 |
| 65 | bge_m3_c256_hybrid | bge-m3 | 256 | 32 | 0.770 | 0.915 | 0.940 | 0.950 | 0.847 | 32.4 |
| 66 | bge_m3_c256_hybrid_rerank | bge-m3 | 256 | 32 | 0.860 | 0.915 | 0.930 | 0.945 | 0.889 | 417.1 |
| 67 | bge_m3_c512_dense | bge-m3 | 512 | 64 | 0.745 | 0.850 | 0.900 | 0.920 | 0.806 | 56.1 |
| 68 | bge_m3_c512_hybrid | bge-m3 | 512 | 64 | 0.785 | 0.905 | 0.930 | 0.960 | 0.849 | 29.7 |
| 69 | bge_m3_c512_hybrid_rerank | bge-m3 | 512 | 64 | 0.845 | 0.925 | 0.935 | 0.950 | 0.884 | 658.8 |
| 70 | user_bge_m3_c1024_dense | USER-bge-m3 | 1024 | 128 | 0.730 | 0.875 | 0.910 | 0.930 | 0.808 | 34.9 |
| 71 | user_bge_m3_c1024_hybrid | USER-bge-m3 | 1024 | 128 | 0.810 | 0.905 | 0.935 | 0.965 | 0.865 | 30.6 |
| 72 | user_bge_m3_c1024_hybrid_rerank | USER-bge-m3 | 1024 | 128 | 0.860 | 0.930 | 0.950 | 0.960 | 0.897 | 1165.8 |
| 73 | user_bge_m3_c2048_dense | USER-bge-m3 | 2048 | 256 | 0.725 | 0.850 | 0.880 | 0.930 | 0.798 | 27.0 |
| 74 | user_bge_m3_c2048_hybrid | USER-bge-m3 | 2048 | 256 | 0.795 | 0.895 | 0.925 | 0.935 | 0.848 | 28.0 |
| 75 | user_bge_m3_c2048_hybrid_rerank | USER-bge-m3 | 2048 | 256 | 0.885 | 0.940 | 0.950 | 0.960 | 0.912 | 2273.3 |
| 76 | user_bge_m3_c256_dense | USER-bge-m3 | 256 | 32 | 0.700 | 0.825 | 0.880 | 0.920 | 0.774 | 30.8 |
| 77 | user_bge_m3_c256_hybrid | USER-bge-m3 | 256 | 32 | 0.790 | 0.915 | 0.925 | 0.940 | 0.853 | 30.9 |
| 78 | user_bge_m3_c256_hybrid_rerank | USER-bge-m3 | 256 | 32 | 0.845 | 0.910 | 0.925 | 0.940 | 0.878 | 411.3 |

*Колонки Faithful и AnsCorr резервируют метрики качества генерации фреймворка RAGAS.*

### Лучший запуск на текущий момент

- **Tag:** `user_bge_m3_c2048_hybrid_rerank`
- **Embedding:** `deepvk/USER-bge-m3`
- **Chunk:** 2048 / overlap 256
- **MRR:** 0.912
- **Hit@1:** 0.885
- **Файл:** `metrics_user_bge_m3_c2048_hybrid_rerank_20261028_032721.json`
