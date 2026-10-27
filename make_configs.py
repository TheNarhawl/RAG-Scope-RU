from pathlib import Path
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# Базовая сетка чанков по умолчанию (chunk_size, chunk_overlap)
DEFAULT_CHUNKS = [
    (256, 32),
    (512, 64),
    (1024, 128),
]

MODELS = {
    # --- Семейство E5 (требуют префиксы passage: и query:) ---
    "e5_small": {
        "hf_name": "intfloat/multilingual-e5-small",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
    "e5_base": {
        "hf_name": "intfloat/multilingual-e5-base",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
    "e5_large": {
        "hf_name": "intfloat/multilingual-e5-large",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },

    # --- Длинноконтекстные и тяжелые мультиязычные модели ---
    "bge_m3": {
        "hf_name": "BAAI/bge-m3",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128), (2048, 256)],
    },
    "gte_multilingual": {
        "hf_name": "Alibaba-NLP/gte-multilingual-base",
        "passage_prefix": "",
        "query_prefix": "",
        # Контекст до 8192 токенов
        "chunks": [(256, 32), (512, 64), (1024, 128), (2048, 256)],
    },

    # --- Адаптации под русский язык (VK, Sber, HSE) ---
    "user_bge_m3": {
        "hf_name": "deepvk/USER-bge-m3",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
    "user_base": {
        "hf_name": "deepvk/USER-base",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
    "rosberta": {
        "hf_name": "ai-forever/ru-en-RoSBERTa",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
    "sbert_large": {
        "hf_name": "ai-forever/sbert_large_nlu_ru",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },

    # --- Легковесные модели и академические бейслайны ---
    "rubert_tiny2": {
        "hf_name": "cointegrated/rubert-tiny2",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64)],
    },
    "labse_ru": {
        "hf_name": "cointegrated/LaBSE-en-ru",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64)],
    },
    "mpnet_multilingual": {
        "hf_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64)],
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
        chunk_list = m_info.get("chunks", DEFAULT_CHUNKS)
        for chunk_size, overlap in chunk_list:
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
                    "chunk_strategy": "structural",
                }

                passage_pref = m_info.get("passage_prefix")
                if passage_pref:
                    ingest_block["passage_prefix"] = passage_pref

                query_pref = m_info.get("query_prefix")
                if query_pref:
                    ingest_block["query_prefix"] = query_pref

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