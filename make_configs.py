from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"

DEFAULT_CHUNKS = [
    (256, 32),
    (512, 64),
    (1024, 128),
]

MODELS = {
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
        "chunks": [(256, 32), (512, 64), (1024, 128), (2048, 256)],
    },

    # --- Адаптации под русский язык (VK, Sber, HSE) ---
    "user_bge_m3": {
        "hf_name": "deepvk/USER-bge-m3",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128), (2048, 256)],
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
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
    "labse_ru": {
        "hf_name": "cointegrated/LaBSE-en-ru",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
    "mpnet_multilingual": {
        "hf_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "passage_prefix": "",
        "query_prefix": "",
        "chunks": [(256, 32), (512, 64), (1024, 128)],
    },
}

STRATEGIES = {
    "sparse": {
        "use_hybrid": False,
        "use_rerank": False,
        "vector_name": "sparse",
    },
    "dense": {
        "use_hybrid": False,
        "use_rerank": False,
        "vector_name": "dense",
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
                    "meta": {
                        "name": run_name,
                        "seed": 42,
                        "device": "auto",
                    },
                    "ingest": ingest_block,
                    "retrieve": retrieve_block,
                }

                out_file = chunk_dir / f"{strat_name}.yaml"
                out_file.write_text(
                    yaml.dump(cfg_data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                count += 1

    # --- A/B тестирование стратегий нарезки (Chunking) ---
    ab_chunk_dir = CONFIGS_DIR / "ab_chunking"
    ab_chunk_dir.mkdir(parents=True, exist_ok=True)

    with open(ab_chunk_dir / "e5_naive.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "meta": {"name": "ab-e5-naive-c512", "seed": 42, "device": "auto"},
                "ingest": {
                    "embedding_model": "intfloat/multilingual-e5-large",
                    "collection": "ab_e5_c512_naive",
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "chunk_strategy": "naive",
                    "passage_prefix": "passage: ",
                    "query_prefix": "query: ",
                },
                "retrieve": {
                    "top_k": 10,
                    "vector_name": "dense",
                    "use_hybrid": True,
                    "use_rerank": False,
                },
            },
            f,
            sort_keys=False,
        )

    with open(ab_chunk_dir / "e5_structural.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "meta": {
                    "name": "ab-e5-structural-c512",
                    "seed": 42,
                    "device": "auto",
                },
                "ingest": {
                    "embedding_model": "intfloat/multilingual-e5-large",
                    "collection": "ab_e5_c512_structural",
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "chunk_strategy": "structural",
                    "passage_prefix": "passage: ",
                    "query_prefix": "query: ",
                },
                "retrieve": {
                    "top_k": 10,
                    "vector_name": "dense",
                    "use_hybrid": True,
                    "use_rerank": False,
                },
            },
            f,
            sort_keys=False,
        )

    # --- A/B тестирование парсеров PDF (pypdf vs pymupdf4llm) по весовым категориям ---
    ab_parse_dir = CONFIGS_DIR / "ab_parsing"
    ab_parse_dir.mkdir(parents=True, exist_ok=True)

    ab_tiers = {
        "small_rubert": {
            "model": "cointegrated/rubert-tiny2",
            "passage_prefix": "",
            "query_prefix": "",
        },
        "base_user_bge": {
            "model": "deepvk/USER-bge-m3",
            "passage_prefix": "",
            "query_prefix": "",
        },
        "large_bge_m3": {
            "model": "BAAI/bge-m3",
            "passage_prefix": "",
            "query_prefix": "",
        },
        "large_e5": {
            "model": "intfloat/multilingual-e5-large",
            "passage_prefix": "passage: ",
            "query_prefix": "query: ",
        },
    }

    ab_parse_count = 0
    for tier_key, tier_info in ab_tiers.items():
        for parser in ["pypdf", "pymupdf4llm"]:
            parser_alias = "pymupdf" if parser == "pymupdf4llm" else "pypdf"
            cfg_name = f"{tier_key}_{parser_alias}"
            run_name = f"ab-parse-{tier_key}-{parser_alias}-c512"
            collection = f"ab_{tier_key}_c512_{parser_alias}"

            cfg_dict = {
                "meta": {"name": run_name, "seed": 42, "device": "auto"},
                "ingest": {
                    "embedding_model": tier_info["model"],
                    "collection": collection,
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "chunk_strategy": "structural",
                    "parser": parser,
                },
                "retrieve": {
                    "top_k": 10,
                    "vector_name": "dense",
                    "use_hybrid": True,
                    "use_rerank": False,
                },
            }
            if tier_info["passage_prefix"]:
                cfg_dict["ingest"]["passage_prefix"] = tier_info["passage_prefix"]
            if tier_info["query_prefix"]:
                cfg_dict["ingest"]["query_prefix"] = tier_info["query_prefix"]

            out_yaml = ab_parse_dir / f"{cfg_name}.yaml"
            out_yaml.write_text(
                yaml.dump(cfg_dict, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            ab_parse_count += 1

    print(
        f"Сгенерировано {count + 2 + ab_parse_count} конфигов (включая A/B тесты) в {CONFIGS_DIR}"
    )


if __name__ == "__main__":
    generate()
