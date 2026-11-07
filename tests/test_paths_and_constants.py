import importlib
import sys
import uuid
from pathlib import Path

# Добавляем src в sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import constants, paths  # noqa: E402


def test_constants_values():
    assert constants.DEFAULT_SEED == 42
    assert constants.DEFAULT_CHUNK_SIZE == 512
    assert constants.DEFAULT_CHUNK_OVERLAP == 64
    assert constants.DEFAULT_MAX_SEQ_LENGTH == 1024
    assert constants.DEFAULT_EMBED_BATCH_SIZE == 32
    assert constants.DEFAULT_SPARSE_MODEL == "Qdrant/bm25"
    assert constants.SPARSE_LANGUAGE == "russian"
    assert constants.DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-base"
    assert constants.DEFAULT_RERANK_CANDIDATES == 20
    assert constants.DEFAULT_TOP_K == 5
    assert constants.DEFAULT_VECTOR_NAME == "dense"
    assert constants.KS == [1, 3, 5, 10]
    assert isinstance(constants.UUID_NAMESPACE, uuid.UUID)
    assert constants.HF_DATASET_URL.startswith("https://")
    assert constants.DEFAULT_PDF_PARSER == "pypdf"
    assert constants.DEFAULT_LLM_MODEL == "deepseek-chat"
    assert "faithfulness" in constants.RAGAS_METRICS
    assert "answer_correctness" in constants.RAGAS_METRICS
    assert "context_recall" in constants.RAGAS_METRICS
    assert "{context}" in constants.DEFAULT_RAG_PROMPT
    assert "{question}" in constants.DEFAULT_RAG_PROMPT


def test_paths_types_and_defaults():
    assert isinstance(paths.ROOT, Path)
    assert paths.ROOT.exists()
    assert paths.CONFIGS_DIR == paths.ROOT / "configs"
    assert paths.DOCS_DIR == paths.DATA_DIR / "docs"
    assert paths.EVAL_DIR == paths.DATA_DIR / "eval"
    assert paths.QDRANT_URL == "http://localhost:6333" or paths.QDRANT_URL.startswith("http")
    assert isinstance(paths.EMBED_BATCH_SIZE, int)


def test_paths_env_override(monkeypatch):
    test_data = "G:/custom_data"
    test_qdrant = "http://qdrant.internal:6333"
    test_batch = "64"

    monkeypatch.setenv("DATA_DIR", test_data)
    monkeypatch.setenv("QDRANT_URL", test_qdrant)
    monkeypatch.setenv("EMBED_BATCH_SIZE", test_batch)

    # Перезагружаем paths для проверки считывания переменных окружения
    reloaded_paths = importlib.reload(paths)
    try:
        assert str(reloaded_paths.DATA_DIR).replace("\\", "/") == test_data
        assert reloaded_paths.QDRANT_URL == test_qdrant
        assert reloaded_paths.EMBED_BATCH_SIZE == 64
    finally:
        # Восстанавливаем дефолты
        monkeypatch.undo()
        importlib.reload(paths)
