import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
from fastembed import SparseTextEmbedding
from qdrant_client import models

from paths import CONFIGS_DIR, EVAL_DIR, EXPERIMENTS_DIR, QDRANT_URL
from utils import (
    get_qdrant_client,
    load_dense_model,
    load_reranker_model,
    pick_device,
    set_seed,
)

KS = [1, 3, 5, 10]
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def slugify(s: str) -> str:
    return s.lower().replace(" ", "_").replace("+", "").replace("-", "_")


def hit_at_k(sources: list[str], target: str, k: int) -> float:
    return 1.0 if target in sources[:k] else 0.0


def rr(sources: list[str], target: str) -> float:
    for i, s in enumerate(sources, start=1):
        if s == target:
            return 1.0 / i
    return 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", default="baseline")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load((CONFIGS_DIR / f"{args.config}.yaml").read_text(encoding="utf-8"))
    seed = cfg.get("meta", {}).get("seed", 42)
    set_seed(seed)

    ing = cfg["ingest"]
    ret = cfg["retrieve"]

    vector_name = ret.get("vector_name", "dense")
    use_hybrid = ret.get("use_hybrid", False)
    use_rerank = ret.get("use_rerank", False)
    rerank_candidates = ret.get("rerank_candidates", 20)
    reranker_model_name = ret.get("reranker_model", DEFAULT_RERANKER_MODEL)

    max_k = max(KS)
    search_limit = max(rerank_candidates, max_k) if use_rerank else max_k

    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    device = pick_device(verbose=False)
    
    # загружаем только те модели, которые нужны для текущего конфига
    dense_model = load_dense_model(ing["embedding_model"], device=device) if (use_hybrid or vector_name == "dense") else None
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian") if (use_hybrid or vector_name == "sparse") else None
    reranker = load_reranker_model(reranker_model_name, device=device) if use_rerank else None
    
    client = get_qdrant_client(QDRANT_URL)

    is_e5 = "e5" in ing["embedding_model"].lower()
    query_prefix = ing.get("query_prefix", "query: " if is_e5 else "")

    info = client.get_collection(ing["collection"])
    print(f"Конфиг: {args.config} (seed={seed}, device={device})")
    print(f"Коллекция: {ing['collection']} ({info.points_count} точек)")
    print(f"Стратегия: {vector_name} | Hybrid: {use_hybrid} | Rerank: {use_rerank} ({reranker_model_name if use_rerank else '—'})")
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
        if use_rerank and hasattr(res, 'points') and res.points:
            pairs = [(qt, h.payload["text"]) for h in res.points]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, res.points), key=lambda x: x[0], reverse=True)
            sources = [h.payload["source"] for _, h in ranked[:max_k]]
        else:
            sources = [h.payload["source"] for h in (res.points if hasattr(res, 'points') else [])[:max_k]]
        total_rerank_ms += (time.time() - t0_rerank) * 1000

        target = q["source_hint"]
        row = {"id": q["id"], "question": qt, "target": target, "retrieved": sources}
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
    
    summary = {
        "n_questions": n,
        "hit_rate": {f"@{k}": hits[k] / n for k in KS},
        "mrr": mrr_sum / n,
        "elapsed_sec": elapsed_global,
        "ms_per_query": avg_total_ms,
        "latency_breakdown_ms": {
            "encode": avg_encode_ms,
            "search": avg_search_ms,
            "rerank": avg_rerank_ms
        }
    }

    print("=== Метрики ===")
    for k in KS:
        print(f"  Hit Rate@{k}: {summary['hit_rate'][f'@{k}']:.3f}")
    print(f"  MRR:         {summary['mrr']:.3f}")
    print(f"  Время (Total): {avg_total_ms:.1f} мс/q")
    print(f"    - Encode:    {avg_encode_ms:.1f} мс/q")
    print(f"    - Search:    {avg_search_ms:.1f} мс/q")
    if use_rerank:
        print(f"    - Rerank:    {avg_rerank_ms:.1f} мс/q")
    print("")

    tag = args.tag or slugify(cfg["meta"]["name"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXPERIMENTS_DIR / ing["collection"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{tag}_{ts}.json"
    
    out_path.write_text(
        json.dumps(
            {
                "config": cfg,
                "tag": tag,
                "timestamp": ts,
                "summary": summary,
                "per_question": per_q,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Сохранено: {out_path}")

    if dense_model: del dense_model
    if sparse_model: del sparse_model
    if reranker: del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()