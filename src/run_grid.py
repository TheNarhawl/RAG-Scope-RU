import argparse
import subprocess
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CONFIGS_DIR, EXPERIMENTS_DIR


def slugify(s: str) -> str:
    return s.lower().replace(" ", "_").replace("+", "").replace("-", "_")


def is_already_done(collection: str, run_name: str) -> bool:
    """Проверяет, есть ли уже JSON с метриками для этого запуска."""
    tag = slugify(run_name)
    col_dir = EXPERIMENTS_DIR / collection
    if not col_dir.exists():
        return False
    return any(col_dir.glob(f"metrics_{tag}_*.json"))


def main():
    parser = argparse.ArgumentParser(description="Пакетный запуск сетки экспериментов")
    parser.add_argument("--filter", type=str, default="", help="Фильтр по названию папки (например, e5_large)")
    parser.add_argument("--force", action="store_true", help="Пересчитать даже если результаты уже есть")
    args = parser.parse_args()

    chunk_dirs = sorted(p.parent for p in CONFIGS_DIR.rglob("dense.yaml") if p.parent != CONFIGS_DIR)

    for c_dir in chunk_dirs:
        rel_dir = c_dir.relative_to(CONFIGS_DIR).as_posix()
        if args.filter and args.filter not in rel_dir:
            continue

        dense_cfg = yaml.safe_load((c_dir / "dense.yaml").read_text(encoding="utf-8"))
        collection = dense_cfg["ingest"]["collection"]

        # Проверяем, какие стратегии нужно посчитать в этой группе
        strats_to_run = []
        for strat in ["dense", "hybrid", "hybrid-rerank"]:
            cfg_file = c_dir / f"{strat}.yaml"
            if cfg_file.exists():
                cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
                run_name = cfg_data["meta"]["name"]
                if args.force or not is_already_done(collection, run_name):
                    strats_to_run.append(strat)

        if not strats_to_run:
            print(f"[ПРОПУСК] Группа {rel_dir} уже полностью посчитана.")
            continue

        print(f"\n==================== ГРУППА: {rel_dir} ====================")
        # 1. Индексация (один раз на группу чанка)
        subprocess.run([sys.executable, "src/ingest.py", "--config", f"{rel_dir}/dense"], check=True)

        # 2. Оценка нужных стратегий
        for strat in strats_to_run:
            subprocess.run([sys.executable, "src/evaluate_retrieval.py", "--config", f"{rel_dir}/{strat}"], check=True)

        # 3. Обновляем таблицу после каждой группы
        subprocess.run([sys.executable, "src/summarize_experiments.py"], check=True)


if __name__ == "__main__":
    main()