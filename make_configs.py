from pathlib import Path
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

MODELS = {
    "rubert_tiny2": {
        "hf_name": "cointegrated/rubert-tiny2",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
        "passage_prefix": "",
        "query_prefix": "",
    },
    "e5_small": {
        "hf_name": "intfloat/multilingual-e5-small",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    },
    "e5_base": {
        "hf_name": "intfloat/multilingual-e5-base",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    },
    "e5_large": {
        "hf_name": "intfloat/multilingual-e5-large",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    },
    "rosberta": {
        "hf_name": "ai-forever/ru-en-RoSBERTa",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
        "passage_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
    "bge_m3": {
        "hf_name": "BAAI/bge-m3",
        "chunks": [(512, 64), (1024, 128), (2048, 256)],
        "passage_prefix": "",
        "query_prefix": "",
    },
    "user_bge_m3": {
        "hf_name": "deepvk/USER-bge-m3",
        "chunks": [(512, 64), (1024, 128), (2048, 256)],
        "passage_prefix": "",
        "query_prefix": "",
    },
}

STRATEGIES = {
    "dense": {
        "use_hybrid": False,
        "use_rerank": False,
    },
    "hybrid": {
        "use_hybrid": True,
        "use_rerank": False,
    },
    "hybrid-rerank": {
        "use_hybrid": True,
        "use_rerank": True,
        "rerank_candidates": 20,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
    },
}


def generate():
    count = 0
    for model_key, m_info in MODELS.items():
        for chunk_size, overlap in m_info["chunks"]:
            chunk_dir = CONFIGS_DIR / model_key / f"c{chunk_size}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            collection_name = f"{model_key}_c{chunk_size}"

            for strat_name, strat_cfg in STRATEGIES.items():
                run_name = f"{model_key}-c{chunk_size}-{strat_name}"
                ingest_block = {
                    "embedding_model": m_info["hf_name"],
                    "collection": collection_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                }
                if m_info["passage_prefix"]:
                    ingest_block["passage_prefix"] = m_info["passage_prefix"]
                if m_info["query_prefix"]:
                    ingest_block["query_prefix"] = m_info["query_prefix"]

                retrieve_block = {
                    "top_k": 10,
                    "vector_name": "dense",
                    **strat_cfg,
                }

                cfg_data = {
                    "meta": {"name": run_name, "device": "auto"},
                    "ingest": ingest_block,
                    "retrieve": retrieve_block,
                }

                out_file = chunk_dir / f"{strat_name}.yaml"
                out_file.write_text(
                    yaml.dump(cfg_data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                count += 1

    print(f"Сгенерировано {count} конфигов в {CONFIGS_DIR}")


if __name__ == "__main__":
    generate()