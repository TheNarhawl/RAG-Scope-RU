import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for p in (str(SRC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml  # noqa: E402
from common.paths import CONFIGS_DIR, EXPERIMENTS_DIR, QDRANT_URL  # noqa: E402
from common.utils import get_qdrant_client  # noqa: E402
from core.metrics import slugify  # noqa: E402

PIPELINES_DIR = Path(__file__).resolve().parent


def is_already_done(collection: str, run_name: str) -> bool:
    """Проверяет, существует ли уже JSON с метриками для данной конфигурации."""
    tag = slugify(run_name)
    col_dir = EXPERIMENTS_DIR / collection
    if not col_dir.exists():
        return False
    return any(col_dir.glob(f"metrics_{tag}_*.json"))


def collection_has_points(collection: str) -> bool:
    """Проверяет, существует ли коллекция в Qdrant и содержит ли она точки."""
    try:
        client = get_qdrant_client(QDRANT_URL)
        info = client.get_collection(collection)
        return (info.points_count or 0) > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Пакетный запуск сетки экспериментов")
    parser.add_argument("--filter", type=str, default="", help="Подстрока для фильтрации папок (например, labse_ru)")
    parser.add_argument("--force", action="store_true", help="Пересчитать заново, даже если JSON уже есть")
    parser.add_argument(
        "--keep-collections", action="store_true",
        help="Не удалять коллекцию из Qdrant после оценки группы",
    )
    args = parser.parse_args()

    chunk_dirs = sorted(p.parent for p in CONFIGS_DIR.rglob("dense.yaml") if p.parent != CONFIGS_DIR)
    failed_groups = []

    ingest_script = str(PIPELINES_DIR / "ingest.py")
    eval_script = str(PIPELINES_DIR / "evaluate.py")
    summary_script = str(PIPELINES_DIR / "summarize_experiments.py")

    for c_dir in chunk_dirs:
        rel_dir = c_dir.relative_to(CONFIGS_DIR).as_posix()
        if args.filter and args.filter not in rel_dir:
            continue

        dense_cfg = yaml.safe_load((c_dir / "dense.yaml").read_text(encoding="utf-8"))
        collection = dense_cfg["ingest"]["collection"]

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
        try:
            # 1. Индексация (пропускаем, если коллекция уже заполнена и нет флага --force)
            if not args.force and collection_has_points(collection):
                print(f"[ИНДЕКС ГОТОВ] Коллекция '{collection}' уже заполнена в Qdrant, пропускаем ingest.py.")
            else:
                subprocess.run(
                    [sys.executable, ingest_script, "--config", f"{rel_dir}/dense"],
                    check=True,
                )

            # 2. Оценка стратегий
            for strat in strats_to_run:
                subprocess.run(
                    [sys.executable, eval_script, "--config", f"{rel_dir}/{strat}"],
                    check=True,
                )

            # 3. Обновление сводной таблицы RESULTS.md
            subprocess.run([sys.executable, summary_script], check=True)

            # 4. Освобождаем память в Qdrant
            if not args.keep_collections:
                try:
                    get_qdrant_client(QDRANT_URL).delete_collection(collection)
                    print(f"Временная коллекция '{collection}' очищена из Qdrant.")
                except Exception:
                    pass

        except subprocess.CalledProcessError as e:
            print(f"\n[ОШИБКА] Группа {rel_dir} завершилась с ошибкой: {e}")
            failed_groups.append(rel_dir)
            continue

    if failed_groups:
        print(f"\nЗавершено с ошибками в группах ({len(failed_groups)}): {failed_groups}")
    else:
        print("\nВсе запланированные группы успешно обработаны!")


if __name__ == "__main__":
    main()
