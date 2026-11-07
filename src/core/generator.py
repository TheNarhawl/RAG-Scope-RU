from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure 'src' and ROOT are on sys.path
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for p in (str(SRC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.constants import DEFAULT_LLM_MODEL, DEFAULT_RAG_PROMPT  # noqa: E402
from common.utils import get_llm_client  # noqa: E402

logger = logging.getLogger(__name__)


def format_rag_prompt(
    question: str,
    contexts: list[str],
    prompt_template: str = DEFAULT_RAG_PROMPT,
) -> str:
    """Форматирует контексты и вопрос в целевой промпт."""
    context_str = "\n\n---\n\n".join(contexts).strip()
    return prompt_template.format(context=context_str, question=question.strip())


def generate_rag_answer(
    question: str,
    contexts: list[str],
    client: Any | None = None,
    model: str = DEFAULT_LLM_MODEL,
    system_prompt: str = DEFAULT_RAG_PROMPT,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Генерирует ответ на основе предоставленных контекстов через OpenAI-совместимый API.
    """
    if client is None:
        client = get_llm_client()

    formatted_prompt = format_rag_prompt(
        question, contexts, prompt_template=system_prompt
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": formatted_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content
        return answer.strip() if answer else ""
    except Exception as e:
        logger.error(f"Ошибка при вызове LLM ({model}): {e}")
        return f"[ERROR: Не удалось сгенерировать ответ: {e}]"
