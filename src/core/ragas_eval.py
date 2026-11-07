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

from common.constants import DEFAULT_LLM_MODEL  # noqa: E402
from common.utils import ensure_ragas_compatibility, get_llm_client  # noqa: E402

logger = logging.getLogger(__name__)


def prepare_ragas_dataset(samples: list[dict[str, Any]]):
    """
    Преобразует список словарей с предсказаниями в HuggingFace Dataset для RAGAS.
    Поддерживает ключи: question / user_input, answer / response,
    contexts / retrieved_contexts, ground_truth / reference.
    """
    from datasets import Dataset

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for s in samples:
        q = s.get("question") or s.get("user_input", "")
        a = s.get("answer") or s.get("response", "")
        c = s.get("contexts") or s.get("retrieved_contexts", [])
        if isinstance(c, str):
            c = [c]
        gt = s.get("ground_truth") or s.get("reference", "")

        questions.append(str(q))
        answers.append(str(a))
        contexts_list.append([str(item) for item in c])
        ground_truths.append(str(gt))

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts_list,
        "reference": ground_truths,
        # Совместимость с legacy ragas версиями
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    return Dataset.from_dict(data)


def evaluate_ragas(
    samples: list[dict[str, Any]],
    client: Any | None = None,
    model: str = DEFAULT_LLM_MODEL,
    metrics: list[Any] | None = None,
    embeddings: Any | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Запускает оценку триады метрик RAGAS:
    - faithfulness (фактическая точность ответа относительно контекста)
    - answer_correctness (соответствие эталонному ответу)
    - context_recall (полнота извлечения контекста для ответа на вопрос)
    """
    if not samples:
        return {}

    ensure_ragas_compatibility()

    try:
        import numpy as np
        from ragas import evaluate
        from ragas.llms import llm_factory
        try:
            from ragas.metrics.collections import (
                AnswerCorrectness,
                ContextRecall,
                Faithfulness,
            )
            use_collections = True
        except ImportError:
            from ragas.metrics import (
                answer_correctness,
                context_recall,
                faithfulness,
            )
            from ragas.metrics._answer_correctness import AnswerCorrectness
            use_collections = False
    except ImportError as e:
        logger.error(f"Не удалось импортировать RAGAS: {e}")
        return {"error": f"RAGAS import failed: {e}", "summary": {}}

    if client is None:
        client = get_llm_client()

    ragas_llm = llm_factory(model=model, client=client)

    if metrics is None:
        if use_collections:
            ac_weights = [1.0, 0.0] if embeddings is None else [0.75, 0.25]
            metrics = [
                Faithfulness(llm=ragas_llm),
                AnswerCorrectness(llm=ragas_llm, weights=ac_weights),
                ContextRecall(llm=ragas_llm),
            ]
        else:
            if embeddings is None:
                ac_metric = AnswerCorrectness(weights=[1.0, 0.0])
            else:
                ac_metric = answer_correctness
            metrics = [faithfulness, ac_metric, context_recall]

    hf_dataset = prepare_ragas_dataset(samples)

    try:
        results = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=embeddings,
            show_progress=show_progress,
            raise_exceptions=False,
        )

        scores: dict[str, float] = {}
        for m in metrics:
            metric_name = getattr(m, "name", str(m))
            val = None
            if hasattr(results, "get"):
                val = results.get(metric_name)
            if val is None and hasattr(results, "__getitem__"):
                try:
                    val = results[metric_name]
                except (KeyError, IndexError):
                    val = None

            if val is not None:
                try:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        val = float(np.nanmean(val))
                    else:
                        val = float(val)
                except Exception:
                    pass
                scores[metric_name] = val

        raw_records = []
        if hasattr(results, "to_pandas"):
            raw_records = results.to_pandas().to_dict(orient="records")

        return {
            "summary": scores,
            "raw": raw_records,
        }
    except Exception as e:
        logger.error(f"Ошибка при выполнении RAGAS evaluate: {e}")
        return {"error": str(e), "summary": {}}
