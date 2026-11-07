from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.generator import format_rag_prompt, generate_rag_answer  # noqa: E402
from core.ragas_eval import evaluate_ragas, prepare_ragas_dataset  # noqa: E402
from pipelines.ingest import load_pdf_pages  # noqa: E402


def test_format_rag_prompt():
    question = "Каковы требования к бетону?"
    contexts = ["Пункт 1: Прочность B25.", "Пункт 2: Морозостойкость F200."]
    prompt = format_rag_prompt(question, contexts)

    assert question in prompt
    assert "Пункт 1: Прочность B25." in prompt
    assert "Пункт 2: Морозостойкость F200." in prompt
    assert "---" in prompt


def test_generate_rag_answer_mocked():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Бетон должен соответствовать классу B25."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    question = "Какой класс бетона?"
    contexts = ["Требуется класс B25."]
    answer = generate_rag_answer(
        question=question,
        contexts=contexts,
        client=mock_client,
        model="deepseek-chat",
    )

    assert answer == "Бетон должен соответствовать классу B25."
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["temperature"] == 0.0


def test_generate_rag_answer_error_handling():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API connection timeout")

    answer = generate_rag_answer(
        question="Тестовый вопрос",
        contexts=["Контекст"],
        client=mock_client,
    )

    assert answer.startswith("[ERROR:")
    assert "API connection timeout" in answer


def test_prepare_ragas_dataset():
    samples = [
        {
            "question": "Вопрос 1",
            "answer": "Ответ 1",
            "contexts": ["Контекст 1A", "Контекст 1B"],
            "ground_truth": "Эталон 1",
        }
    ]
    ds = prepare_ragas_dataset(samples)

    assert "user_input" in ds.column_names
    assert "response" in ds.column_names
    assert "retrieved_contexts" in ds.column_names
    assert "reference" in ds.column_names
    assert ds[0]["user_input"] == "Вопрос 1"
    assert ds[0]["response"] == "Ответ 1"
    assert ds[0]["retrieved_contexts"] == ["Контекст 1A", "Контекст 1B"]
    assert ds[0]["reference"] == "Эталон 1"


def test_evaluate_ragas_empty():
    res = evaluate_ragas([])
    assert res == {}


def test_evaluate_ragas_mocked():
    samples = [
        {
            "question": "Вопрос",
            "answer": "Ответ",
            "contexts": ["Контекст"],
            "ground_truth": "Эталон",
        }
    ]
    mock_client = MagicMock()

    mock_eval_result = MagicMock()
    mock_eval_result.get.side_effect = lambda k: {
        "faithfulness": 0.95,
        "answer_correctness": 0.88,
        "context_recall": 0.91,
    }.get(k)
    mock_eval_result.to_pandas.return_value = MagicMock(
        to_dict=lambda orient: [{"faithfulness": 0.95}]
    )

    from ragas.llms.base import InstructorBaseRagasLLM
    mock_llm = MagicMock(spec=InstructorBaseRagasLLM)

    with patch("ragas.evaluate", return_value=mock_eval_result), patch(
        "ragas.llms.llm_factory"
    ) as mock_llm_factory:
        mock_llm_factory.return_value = mock_llm
        res = evaluate_ragas(samples, client=mock_client)

        assert "summary" in res
        summary = res["summary"]
        assert summary.get("faithfulness") == 0.95
        assert summary.get("answer_correctness") == 0.88
        assert summary.get("context_recall") == 0.91


def test_load_pdf_pages_parsers(tmp_path):
    # Тестируем fallback на pypdf при ошибке pymupdf4llm
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n%EOF")

    with patch("pypdf.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Текст документа ГОСТ"
        mock_reader.return_value.pages = [mock_page]

        pages = load_pdf_pages(dummy_pdf, parser="pypdf")
        assert len(pages) == 1
        assert pages[0] == (1, "Текст документа ГОСТ")
