import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.splitters import naive_chunk_text, structural_chunk_text  # noqa: E402


def test_empty_text():
    """Пустая строка должна возвращать пустой список."""
    assert structural_chunk_text("") == []
    assert structural_chunk_text("   \n  ") == []


def test_short_text_no_split():
    """Текст меньше размера чанка не должен разбиваться."""
    text = "Короткий текст ГОСТ."
    result = structural_chunk_text(text, chunk_size=512)
    assert result == [text]


def test_structural_prefix_preserved():
    """Структурные маркеры (Статья, Глава) должны сохраняться в начале чанка."""
    text = "\nСтатья 1. Термины и определения.\nСтатья 2. Общие положения."
    result = structural_chunk_text(text, chunk_size=40, chunk_overlap=0)
    assert len(result) == 2
    assert result[0] == "Статья 1. Термины и определения."
    assert result[1] == "Статья 2. Общие положения."


def test_sentence_punctuation_preserved():
    """При разбиении по предложениям знаки препинания должны оставаться в конце."""
    text = "Первое предложение. Второе предложение; Третье предложение."
    result = structural_chunk_text(text, chunk_size=25, chunk_overlap=0)
    assert len(result) >= 2
    assert "Первое предложение." in result[0]
    assert any("Второе предложение;" in r for r in result)


def test_large_block_recursive_split():
    """Слишком длинный блок без абзацев должен рекурсивно биться по предложениям."""
    text = "\nСтатья 1. " + "Очень длинное предложение. " * 10
    result = structural_chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(result) > 1
    assert result[0].startswith("Статья 1.")
    for r in result:
        assert len(r) <= 120


def test_overlap_logic_with_sentences():
    """Проверка корректного формирования overlap на уровне предложений."""
    text = "Первое. Второе. Третье."
    result = structural_chunk_text(text, chunk_size=18, chunk_overlap=10)

    assert len(result) == 2
    assert result[0] == "Первое. Второе."
    assert "Второе." in result[1]
    assert "Третье." in result[1]


def test_fallback_character_split():
    """Если нет никаких разделителей, текст должен успешно разбиваться с перекрытием."""
    text = "А" * 100
    result = structural_chunk_text(text, chunk_size=30, chunk_overlap=5)

    assert len(result) > 1
    for r in result:
        assert len(r) > 0


def test_real_gost_simulation():
    """Симуляция реального фрагмента нормативного документа."""
    text = (
        "\nРаздел 1. Область применения\n"
        "Настоящий стандарт распространяется на...\n"
        "\nРаздел 2. Нормативные ссылки\n"
        "В настоящем стандарте использованы нормативные ссылки на следующие документы:\n"
        "ГОСТ Р ИСО 9001-2015 Системы менеджмента качества.\n"
        "\nПункт 2.1. Особые условия."
    )
    result = structural_chunk_text(text, chunk_size=120, chunk_overlap=20)

    assert len(result) > 1
    assert result[0].startswith("Раздел 1.")
    assert any("Раздел 2. Нормативные ссылки" in r for r in result)
    assert any(r.startswith("Пункт 2.1.") for r in result)


def test_naive_chunk_empty():
    assert naive_chunk_text("", size=100, overlap=20) == []
    assert naive_chunk_text("   ", size=100, overlap=20) == []


def test_naive_chunk_short_text():
    text = "Короткий текст"
    assert naive_chunk_text(text, size=100, overlap=20) == [text]


def test_naive_chunk_overlap():
    text = "1234567890abcdefghij"
    chunks = naive_chunk_text(text, size=10, overlap=2)
    assert len(chunks) >= 2
    assert chunks[0] == "1234567890"
    # Следующий шаг start = 10 - 2 = 8
    assert chunks[1] == "90abcdefgh"


def test_markdown_header_splitting():
    """Проверка разбиения по Markdown-заголовкам (#, ##, ###) от pymupdf4llm."""
    text = (
        "# Заголовок 1\nТекст первой секции стандарта.\n"
        "## Подраздел 1.1\nДетализированные требования.\n"
        "### Пункт 1.1.1\nКонкретные числовые параметры."
    )
    result = structural_chunk_text(text, chunk_size=55, chunk_overlap=0)
    assert len(result) >= 3
    assert result[0].startswith("# Заголовок 1")
    assert any(r.startswith("## Подраздел 1.1") for r in result)
    assert any("### Пункт 1.1.1" in r for r in result)


def test_markdown_table_splitting():
    """Проверка сохранения структуры Markdown-таблиц."""
    text = (
        "# Таблица параметров\n"
        "| Параметр | Значение |\n"
        "| --- | --- |\n"
        "| Давление | 10 МПа |\n"
        "| Температура | 250 C |"
    )
    result = structural_chunk_text(text, chunk_size=200, chunk_overlap=0)
    assert len(result) == 1
    assert "| Параметр | Значение |" in result[0]
