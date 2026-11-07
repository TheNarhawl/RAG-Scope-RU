import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for p in (str(SRC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
    import yaml
    from fastembed import SparseTextEmbedding
    from qdrant_client import models
    from common.paths import (
        CONFIGS_DIR,
        EVAL_DIR,
        EXPERIMENTS_DIR,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        QDRANT_URL,
        RAGAS_LLM_MODEL,
    )
    from common.utils import (
        get_llm_client,
        get_qdrant_client,
        load_dense_model,
        load_reranker_model,
        pick_device,
        set_seed,
    )
    from core.generator import generate_rag_answer
    from core.metrics import hit_at_k, rr, slugify
    from core.ragas_eval import evaluate_ragas
except ImportError:
    torch = None
    yaml = None
    SparseTextEmbedding = None
    models = None

from common.constants import (  # noqa: E402
    DEFAULT_LLM_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_SEED,
    DEFAULT_SPARSE_MODEL,
    DEFAULT_VECTOR_NAME,
    KS,
    SPARSE_LANGUAGE,
)

__all__ = ["hit_at_k", "rr", "slugify", "KS", "DEFAULT_RERANKER_MODEL"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", default="baseline")
    p.add_argument(
        "--with-ragas",
        action="store_true",
        help="Запустить Stage 2 (генерация LLM + метрики RAGAS)",
    )
    p.add_argument(
        "--ragas-limit",
        type=int,
        default=None,
        help="Ограничение числа вопросов для RAGAS (экономия токенов)",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="Модель LLM для генерации и RAGAS (по умолчанию deepseek-chat)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load((CONFIGS_DIR / f"{args.config}.yaml").read_text(encoding="utf-8"))
    seed = cfg.get("meta", {}).get("seed", DEFAULT_SEED)
    set_seed(seed)

    ing = cfg["ingest"]
    ret = cfg["retrieve"]

    vector_name = ret.get("vector_name", DEFAULT_VECTOR_NAME)
    use_hybrid = ret.get("use_hybrid", False)
    use_rerank = ret.get("use_rerank", False)
    rerank_candidates = ret.get("rerank_candidates", DEFAULT_RERANK_CANDIDATES)
    reranker_model_name = ret.get("reranker_model", DEFAULT_RERANKER_MODEL)
    llm_model = args.llm_model or RAGAS_LLM_MODEL or DEFAULT_LLM_MODEL

    max_k = max(KS)
    search_limit = max(rerank_candidates, max_k) if use_rerank else max_k

    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    device = pick_device(verbose=False)

    # загружаем только те модели, которые нужны для текущего конфига
    dense_model = (
        load_dense_model(ing["embedding_model"], device=device)
        if (use_hybrid or vector_name == "dense") else None
    )
    sparse_model = (
        SparseTextEmbedding(model_name=DEFAULT_SPARSE_MODEL, language=SPARSE_LANGUAGE)
        if (use_hybrid or vector_name == "sparse") else None
    )
    reranker = (
        load_reranker_model(reranker_model_name, device=device)
        if use_rerank else None
    )

    client = get_qdrant_client(QDRANT_URL)

    is_e5 = "e5" in ing["embedding_model"].lower()
    query_prefix = ing.get("query_prefix", "query: " if is_e5 else "")

    info = client.get_collection(ing["collection"])
    print(f"Конфиг: {args.config} (seed={seed}, device={device})")
    print(f"Коллекция: {ing['collection']} ({info.points_count} точек)")
    rerank_display = reranker_model_name if use_rerank else "—"
    print(f"Стратегия: {vector_name} | Hybrid: {use_hybrid} | Rerank: {use_rerank} ({rerank_display})")
    print(f"Вопросов: {len(questions)}\n")

    hits = {k: 0.0 for k in KS}
    mrr_sum = 0.0
    per_q = []

    total_encode_ms = 0.0
    total_search_ms = 0.0
    total_rerank_ms = 0.0

    t0_global = time.time()

    for q in questions:
        qt = q["question"]

        # 1. Замер времени векторизации текста (Encode Latency)
        t0_encode = time.time()
        q_dense = None
        q_sparse = None

        if dense_model:
            q_dense_text = query_prefix + qt if query_prefix else qt
            q_dense = dense_model.encode([q_dense_text], normalize_embeddings=True)[0].tolist()

        if sparse_model:
            q_sparse = list(sparse_model.embed([qt]))[0]

        total_encode_ms += (time.time() - t0_encode) * 1000

        # 2. Замер сетевой задержки и поиска в Qdrant (Search Latency)
        t0_search = time.time()
        if use_hybrid:
            if len(q_sparse.indices) > 0:
                res = client.query_points(
                    collection_name=ing["collection"],
                    prefetch=[
                        models.Prefetch(
                            query=q_dense,
                            using="dense",
                            limit=search_limit,
                        ),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=q_sparse.indices.tolist(),
                                values=q_sparse.values.tolist(),
                            ),
                            using="sparse",
                            limit=search_limit,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=search_limit,
                    with_payload=True,
                )
            else:
                # Защита на случай, если BM25 не нашел ни одного токена
                res = client.query_points(
                    collection_name=ing["collection"],
                    query=q_dense,
                    using="dense",
                    limit=search_limit,
                    with_payload=True,
                )
        elif vector_name == "sparse":
            # Чистый BM25 поиск
            if len(q_sparse.indices) > 0:
                res = client.query_points(
                    collection_name=ing["collection"],
                    query=models.SparseVector(
                        indices=q_sparse.indices.tolist(),
                        values=q_sparse.values.tolist(),
                    ),
                    using="sparse",
                    limit=search_limit,
                    with_payload=True,
                )
            else:
                res = models.ScoredPoint(points=[])
        else:
            # Классический Dense поиск
            res = client.query_points(
                collection_name=ing["collection"],
                query=q_dense,
                using="dense",
                limit=search_limit,
                with_payload=True,
            )
        total_search_ms += (time.time() - t0_search) * 1000

        t0_rerank = time.time()
        if use_rerank and hasattr(res, "points") and res.points:
            pairs = [(qt, h.payload["text"]) for h in res.points]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, res.points), key=lambda x: x[0], reverse=True)
            selected_points = [h for _, h in ranked[:max_k]]
        else:
            selected_points = (res.points if hasattr(res, "points") else [])[:max_k]

        sources = [h.payload["source"] for h in selected_points]
        retrieved_texts = [
            h.payload["text"] for h in selected_points if "text" in h.payload
        ]
        total_rerank_ms += (time.time() - t0_rerank) * 1000

        target = q["source_hint"]
        row = {
            "id": q["id"],
            "question": qt,
            "target": target,
            "ground_truth": q.get("ground_truth", ""),
            "retrieved": sources,
            "retrieved_texts": retrieved_texts,
        }
        for k in KS:
            h = hit_at_k(sources, target, k)
            hits[k] += h
            row[f"hit@{k}"] = h
        r = rr(sources, target)
        mrr_sum += r
        row["rr"] = r
        per_q.append(row)

    elapsed_global = time.time() - t0_global
    n = len(questions)

    avg_encode_ms = total_encode_ms / n
    avg_search_ms = total_search_ms / n
    avg_rerank_ms = total_rerank_ms / n if use_rerank else 0.0
    avg_total_ms = avg_encode_ms + avg_search_ms + avg_rerank_ms

    ragas_summary = {}
    ragas_raw = []
    if args.with_ragas:
        print("\n=== Stage 2: Генерация ответов и оценка RAGAS ===")
        ragas_questions = per_q[: args.ragas_limit] if args.ragas_limit else per_q
        print(f"Оценка на {len(ragas_questions)} вопросах (LLM: {llm_model})...")

        llm_client = get_llm_client(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        ragas_samples = []

        for idx, item in enumerate(ragas_questions, start=1):
            q_text = item["question"]
            ctxs = item["retrieved_texts"]
            gt = item["ground_truth"]
            print(f"[{idx}/{len(ragas_questions)}] Генерация: {q_text[:50]}...")
            gen_ans = generate_rag_answer(
                question=q_text,
                contexts=ctxs,
                client=llm_client,
                model=llm_model,
            )
            item["generated_answer"] = gen_ans
            ragas_samples.append({
                "question": q_text,
                "answer": gen_ans,
                "contexts": ctxs,
                "ground_truth": gt,
            })

        print("Вычисление метрик RAGAS (faithfulness, answer_correctness, context_recall)...")
        ragas_result = evaluate_ragas(
            samples=ragas_samples,
            client=llm_client,
            model=llm_model,
            show_progress=True,
        )
        ragas_summary = ragas_result.get("summary", {})
        ragas_raw = ragas_result.get("raw", [])

    summary = {
        "n_questions": n,
        "hit_rate": {f"@{k}": hits[k] / n for k in KS},
        "mrr": mrr_sum / n,
        "elapsed_sec": elapsed_global,
        "ms_per_query": avg_total_ms,
        "latency_breakdown_ms": {
            "encode": avg_encode_ms,
            "search": avg_search_ms,
            "rerank": avg_rerank_ms,
        },
    }
    if ragas_summary:
        summary["ragas"] = ragas_summary

    print("\n=== Метрики ===")
    for k in KS:
        print(f"  Hit Rate@{k}: {summary['hit_rate'][f'@{k}']:.3f}")
    print(f"  MRR:         {summary['mrr']:.3f}")
    print(f"  Время (Total): {avg_total_ms:.1f} мс/q")
    print(f"    - Encode:    {avg_encode_ms:.1f} мс/q")
    print(f"    - Search:    {avg_search_ms:.1f} мс/q")
    if use_rerank:
        print(f"    - Rerank:    {avg_rerank_ms:.1f} мс/q")

    if ragas_summary:
        print("=== RAGAS Метрики ===")
        if "faithfulness" in ragas_summary:
            print(f"  Faithfulness:       {ragas_summary['faithfulness']:.3f}")
        if "answer_correctness" in ragas_summary:
            print(f"  Answer Correctness: {ragas_summary['answer_correctness']:.3f}")
        if "context_recall" in ragas_summary:
            print(f"  Context Recall:     {ragas_summary['context_recall']:.3f}")
    print("")

    tag = args.tag or slugify(cfg["meta"]["name"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXPERIMENTS_DIR / ing["collection"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{tag}_{ts}.json"

    payload_to_save = {
        "config": cfg,
        "tag": tag,
        "timestamp": ts,
        "summary": summary,
        "per_question": per_q,
    }
    if ragas_raw:
        payload_to_save["ragas_raw"] = ragas_raw

    out_path.write_text(
        json.dumps(
            payload_to_save,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Сохранено: {out_path}")

    if dense_model:
        del dense_model
    if sparse_model:
        del sparse_model
    if reranker:
        del reranker
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
