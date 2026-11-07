import uuid

# --- Воспроизводимость и генераторы ---
DEFAULT_SEED = 42

# --- Параметры чанкинга и последовательностей ---
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_EMBED_BATCH_SIZE = 32

# --- Модели и параметры поиска ---
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
SPARSE_LANGUAGE = "russian"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANK_CANDIDATES = 20
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_NAME = "dense"

# --- Метрики ранжирования ---
KS = [1, 3, 5, 10]

# --- Детерминированные идентификаторы точек Qdrant ---
UUID_NAMESPACE = uuid.UUID("6f2a3b1c-4d5e-4f6a-8b7c-9d0e1f2a3b4c")

# --- Датасет ---
HF_DATASET_URL = (
    "https://huggingface.co/datasets/FractalGPT/RRNCBPublic/resolve/main/rrncb_public_dataset.csv"
)

# --- Парсинг документов ---
DEFAULT_PDF_PARSER = "pypdf"

# --- RAGAS и генерация ответов ---
DEFAULT_LLM_MODEL = "deepseek-chat"
RAGAS_METRICS = ["faithfulness", "answer_correctness", "context_recall"]

DEFAULT_RAG_PROMPT = (
    "Вы — экспертный аналитический ассистент по нормативно-технической и "
    "строительной документации РФ (ГОСТ, СНиП, СП).\n"
    "Ответьте на вопрос пользователя, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленный контекст.\n"
    "Если контекст не содержит ответа или информации недостаточно, прямо укажите: "
    '"Информация в предоставленных документах отсутствует."\n'
    "Не придумывайте факты, числовые параметры и требования от себя. Будьте точны и кратки.\n\n"
    "Контекст:\n{context}\n\n"
    "Вопрос: {question}\n\n"
    "Ответ:"
)
