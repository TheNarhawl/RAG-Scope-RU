import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for p in (str(SRC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import yaml
    from fastembed import SparseTextEmbedding
    from qdrant_client import models
except ImportError:
    yaml = None
    SparseTextEmbedding = None
    models = None

from common.constants import (  # noqa: E402
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_SEED,
    DEFAULT_SPARSE_MODEL,
    DEFAULT_VECTOR_NAME,
    SPARSE_LANGUAGE,
)
from common.paths import CONFIGS_DIR, QDRANT_URL  # noqa: E402
from common.utils import (  # noqa: E402
    get_qdrant_client,
    load_dense_model,
    load_reranker_model,
    pick_device,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline")
    p.add_argument("query", nargs="*")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load((CONFIGS_DIR / f"{args.config}.yaml").read_text(encoding="utf-8"))
    set_seed(cfg.get("meta", {}).get("seed", DEFAULT_SEED))

    ing = cfg["ingest"]
    ret = cfg["retrieve"]

    query = " ".join(args.query) if args.query else "о чём этот документ?"
    top_k = ret["top_k"]
    vector_name = ret.get("vector_name", DEFAULT_VECTOR_NAME)
    use_hybrid = ret.get("use_hybrid", False)
    use_rerank = ret.get("use_rerank", False)
    rerank_candidates = ret.get("rerank_candidates", DEFAULT_RERANK_CANDIDATES)
    reranker_model_name = ret.get("reranker_model", DEFAULT_RERANKER_MODEL)

    device = pick_device(verbose=False)
    dense_model = load_dense_model(ing["embedding_model"], device=device)
    client = get_qdrant_client(QDRANT_URL)

    is_e5 = "e5" in ing["embedding_model"].lower()
    query_prefix = ing.get("query_prefix", "query: " if is_e5 else "")
    q_dense_text = query_prefix + query if query_prefix else query
    q_dense = dense_model.encode([q_dense_text], normalize_embeddings=True)[0].tolist()

    if use_hybrid:
        sparse_model = SparseTextEmbedding(model_name=DEFAULT_SPARSE_MODEL, language=SPARSE_LANGUAGE)
        q_sparse = list(sparse_model.embed([query]))[0]
        if len(q_sparse.indices) > 0:
            results = client.query_points(
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
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=rerank_candidates if use_rerank else top_k,
                with_payload=True,
            )
        else:
            results = client.query_points(
                collection_name=ing["collection"],
                query=q_dense,
                using=vector_name,
                limit=rerank_candidates if use_rerank else top_k,
                with_payload=True,
            )
    else:
        results = client.query_points(
            collection_name=ing["collection"],
            query=q_dense,
            using=vector_name,
            limit=rerank_candidates if use_rerank else top_k,
            with_payload=True,
        )

    hits = results.points

    if use_rerank and hits:
        reranker = load_reranker_model(reranker_model_name, device=device)
        pairs = [(query, h.payload.get("text", "")) for h in hits if h.payload]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        final = [(s, h) for s, h in ranked[:top_k]]
    else:
        final = [(h.score, h) for h in hits[:top_k]]

    print(f"Конфиг: {args.config}")
    print(f"Hybrid: {use_hybrid} | Rerank: {use_rerank}")
    print(f"Запрос: {query}\n")
    for score, h in final:
        p = h.payload
        print(f"--- {p['source']}, стр. {p['page']} (chunk {p['chunk_id']}) score={score:.4f} ---")
        print(p["text"][:400], "\n")


if __name__ == "__main__":
    main()
