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

    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    device = pick_device(verbose=False)
    dense_model = load_dense_model(ing["embedding_model"], device=device)
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian") if use_hybrid else None
    reranker = load_reranker_model(reranker_model_name, device=device) if use_rerank else None
    client = get_qdrant_client(QDRANT_URL)

    is_e5 = "e5" in ing["embedding_model"].lower()
    query_prefix = ing.get("query_prefix", "query: " if is_e5 else "")

    info = client.get_collection(ing["collection"])
    print(f"Конфиг: {args.config} (seed={seed}, device={device})")
    print(f"Коллекция: {ing['collection']} ({info.points_count} точек)")
    print(f"Hybrid: {use_hybrid} | Rerank: {use_rerank} ({reranker_model_name if use_rerank else '—'})")
    print(f"Вопросов: {len(questions)}\n")

    max_k = max(KS)
    hits = {k: 0.0 for k in KS}
    mrr_sum = 0.0
    per_q = []
    t0 = time.time()

    for q in questions:
        qt = q["question"]
        q_dense_text = query_prefix + qt if query_prefix else qt
        q_dense = dense_model.encode([q_dense_text], normalize_embeddings=True)[0].tolist()

        if use_hybrid:
            q_sparse = list(sparse_model.embed([qt]))[0]
            res = client.query_points(
                collection_name=ing["collection"],
                prefetch=[
                    models.Prefetch(query=q_dense, using=vector_name, limit=rerank_candidates),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=q_sparse.indices.tolist(),
                            values=q_sparse.values.tolist(),
                        ),
                        using="sparse",
                        limit=rerank_candidates,
                    ),
                ],
                query=models.RrfQuery(rrf=models.Rrf()),
                limit=rerank_candidates if use_rerank else max_k,
                with_payload=True,
            )
        else:
            res = client.query_points(
                collection_name=ing["collection"],
                query=q_dense,
                using=vector_name,
                limit=rerank_candidates if use_rerank else max_k,
                with_payload=True,
            )

        if use_rerank and res.points:
            pairs = [(qt, h.payload["text"]) for h in res.points]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, res.points), key=lambda x: x[0], reverse=True)
            sources = [h.payload["source"] for _, h in ranked[:max_k]]
        else:
            sources = [h.payload["source"] for h in res.points[:max_k]]

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

    elapsed = time.time() - t0
    n = len(questions)
    summary = {
        "n_questions": n,
        "hit_rate": {f"@{k}": hits[k] / n for k in KS},
        "mrr": mrr_sum / n,
        "elapsed_sec": elapsed,
        "ms_per_query": elapsed / n * 1000,
    }

    print("=== Метрики ===")
    for k in KS:
        print(f"  Hit Rate@{k}: {summary['hit_rate'][f'@{k}']:.3f}")
    print(f"  MRR:         {summary['mrr']:.3f}")
    print(f"  Время:       {elapsed:.1f} сек ({summary['ms_per_query']:.1f} мс/вопрос)\n")

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

    del dense_model
    if reranker is not None:
        del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()