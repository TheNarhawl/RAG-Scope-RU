import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.summarize_experiments import (  # noqa: E402
    build_markdown,
    fmt_any,
    fmt_float,
    fmt_ms,
    short_model,
)


def test_fmt_helpers():
    assert fmt_float(None) == "—"
    assert fmt_float(0.12345) == "0.123"
    assert fmt_float("0.5") == "0.5"

    assert fmt_ms(None) == "—"
    assert fmt_ms(15.678) == "15.7"

    assert fmt_any(None) == "—"
    assert fmt_any(100) == "100"


def test_short_model():
    assert short_model("sentence-transformers/all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"
    assert short_model("deepvk/USER-bge-m3") == "USER-bge-m3"
    assert short_model("intfloat/multilingual-e5-small") == "multilingual-e5-small"
    assert short_model("custom-model") == "custom-model"


def test_build_markdown_empty():
    md = build_markdown([])
    assert "# Experiment results" in md
    assert "Всего прогонов: **0**" in md
    assert "_Пока нет ни одного прогона._" in md


def test_build_markdown_with_runs():
    mock_runs = [
        {
            "file": "metrics_01.json",
            "collection": "test_collection",
            "embedding": "deepvk/USER-bge-m3",
            "chunk_size": 512,
            "overlap": 64,
            "tag": "run_test",
            "timestamp": "2026-11-07 12:00:00",
            "n": 50,
            "hit@1": 0.8,
            "hit@3": 0.9,
            "hit@5": 0.95,
            "hit@10": 1.0,
            "mrr": 0.852,
            "ms_per_query": 14.2,
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
        }
    ]

    md = build_markdown(mock_runs)
    assert "# Experiment results" in md
    assert "Всего прогонов: **1**" in md
    assert "USER-bge-m3" in md
    assert "0.852" in md
    assert "14.2" in md
    assert "## Best run (по MRR)" in md
    assert "**Tag:** `run_test`" in md
