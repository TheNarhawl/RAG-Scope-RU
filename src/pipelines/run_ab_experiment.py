import argparse
import json
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
    tag = slugify(run_name)
    col_dir = EXPERIMENTS_DIR / collection
    if not col_dir.exists():
        return False
    return any(col_dir.glob(f"metrics_{tag}_*.json"))


def collection_has_points(collection: str) -> bool:
    try:
        client = get_qdrant_client(QDRANT_URL)
        info = client.get_collection(collection)
        return (info.points_count or 0) > 0
    except Exception:
        return False


def get_latest_metrics(collection: str, run_name: str) -> dict | None:
    tag = slugify(run_name)
    col_dir = EXPERIMENTS_DIR / collection
    if not col_dir.exists():
        return None
    matches = sorted(col_dir.glob(f"metrics_{tag}_*.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def print_comparison_table(results_by_pair: dict):
    print("\n" + "=" * 95)
    print("                      ИТОГИ A/B ЭКСПЕРИМЕНТА: PYPDF vs PYMUPDF4LLM")
    print("=" * 95)
    header = (
        f"{'Пара/Модель':<20} | {'Парсер':<11} | {'Hit@1':<7} | {'Hit@5':<7} | "
        f"{'MRR':<7} | {'CntxRec':<7} | {'Faithful':<8} | {'ms/q':<6}"
    )
    print(header)
    print("-" * 95)

    for pair_name, variants in results_by_pair.items():
        pypdf_res = variants.get("pypdf", {}).get("summary", {})
        pymupdf_res = variants.get("pymupdf", {}).get("summary", {})

        def get_stat(summary, key, subkey=None):
            if not summary:
                return None
            if subkey:
                val = summary.get(key, {}).get(subkey)
            else:
                val = summary.get(key)
            return val

        h1_py = get_stat(pypdf_res, "hit_rate", "@1")
        h1_mu = get_stat(pymupdf_res, "hit_rate", "@1")
        h5_py = get_stat(pypdf_res, "hit_rate", "@5")
        h5_mu = get_stat(pymupdf_res, "hit_rate", "@5")
        mrr_py = get_stat(pypdf_res, "mrr")
        mrr_mu = get_stat(pymupdf_res, "mrr")
        cr_py = get_stat(pypdf_res, "ragas", "context_recall")
        cr_mu = get_stat(pymupdf_res, "ragas", "context_recall")
        f_py = get_stat(pypdf_res, "ragas", "faithfulness")
        f_mu = get_stat(pymupdf_res, "ragas", "faithfulness")
        ms_py = get_stat(pypdf_res, "ms_per_query")
        ms_mu = get_stat(pymupdf_res, "ms_per_query")

        def f3(val):
            return f"{val:.3f}" if isinstance(val, (int, float)) else "—"

        def f1(val):
            return f"{val:.1f}" if isinstance(val, (int, float)) else "—"

        print(
            f"{pair_name:<20} | {'pypdf':<11} | {f3(h1_py):<7} | {f3(h5_py):<7} | "
            f"{f3(mrr_py):<7} | {f3(cr_py):<7} | {f3(f_py):<8} | {f1(ms_py):<6}"
        )
        print(
            f"{'':<20} | {'pymupdf4llm':<11} | {f3(h1_mu):<7} | {f3(h5_mu):<7} | "
            f"{f3(mrr_mu):<7} | {f3(cr_mu):<7} | {f3(f_mu):<8} | {f1(ms_mu):<6}"
        )

        def diff_str(v1, v2):
            if v1 is None or v2 is None:
                return "—"
            delta = v2 - v1
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.3f}"

        print(
            f"{'':<20} | {'ДЕЛЬТА':<11} | {diff_str(h1_py, h1_mu):<7} | {diff_str(h5_py, h5_mu):<7} | "
            f"{diff_str(mrr_py, mrr_mu):<7} | {diff_str(cr_py, cr_mu):<7} | {diff_str(f_py, f_mu):<8} | "
            f"{diff_str(ms_py, ms_mu):<6}"
        )
        print("-" * 95)


def main():
    parser = argparse.ArgumentParser(description="Запуск A/B тестирования (например, парсеров PDF)")
    parser.add_argument("--group", type=str, default="ab_parsing", help="Имя папки с A/B конфигами")
    parser.add_argument(
        "--pair", "--filter",
        dest="pair",
        type=str,
        default=None,
        help="Фильтр по имени пары/модели (например: large_e5, small_rubert, large_bge)",
    )
    parser.add_argument("--with-ragas", action="store_true", help="Запустить генерацию и расчет RAGAS")
    parser.add_argument("--ragas-limit", type=int, default=None, help="Лимит вопросов для RAGAS")
    parser.add_argument("--llm-model", type=str, default=None, help="Модель для RAGAS")
    parser.add_argument("--force", action="store_true", help="Пересчитать заново")
    parser.add_argument("--keep-collections", action="store_true", help="Не удалять коллекции из Qdrant")
    args = parser.parse_args()

    group_dir = CONFIGS_DIR / args.group
    if not group_dir.exists():
        print(f"Папка {group_dir} не найдена.")
        return

    yaml_files = sorted(group_dir.glob("*.yaml"))
    if args.pair:
        yaml_files = [y for y in yaml_files if args.pair.lower() in y.stem.lower()]
        if not yaml_files:
            print(f"По фильтру '{args.pair}' конфигов не найдено в {group_dir}.")
            return

    if not yaml_files:
        print(f"В папке {group_dir} нет .yaml файлов.")
        return

    ingest_script = str(PIPELINES_DIR / "ingest.py")
    eval_script = str(PIPELINES_DIR / "evaluate.py")
    summary_script = str(PIPELINES_DIR / "summarize_experiments.py")

    results_by_pair = {}

    for y_file in yaml_files:
        rel_cfg = f"{args.group}/{y_file.stem}"
        cfg_data = yaml.safe_load(y_file.read_text(encoding="utf-8"))
        collection = cfg_data["ingest"]["collection"]
        run_name = cfg_data["meta"]["name"]
        parser_type = "pymupdf" if "pymupdf" in y_file.stem else "pypdf"

        # Определяем пару (например, small_rubert, base_user_bge, etc.)
        pair_key = y_file.stem.replace("_pymupdf", "").replace("_pypdf", "")
        if pair_key not in results_by_pair:
            results_by_pair[pair_key] = {}

        if not args.force and is_already_done(collection, run_name):
            print(f"[ПРОПУСК] {rel_cfg} уже посчитан.")
            results_by_pair[pair_key][parser_type] = get_latest_metrics(collection, run_name)
            continue

        print(f"\n==================== ЗАПУСК A/B: {rel_cfg} ====================")
        # 1. Ingest
        if not args.force and collection_has_points(collection):
            print(f"[ИНДЕКС ГОТОВ] Коллекция '{collection}' уже заполнена.")
        else:
            subprocess.run([sys.executable, ingest_script, "--config", rel_cfg], check=True)

        # 2. Evaluate
        eval_cmd = [sys.executable, eval_script, "--config", rel_cfg]
        if args.with_ragas:
            eval_cmd.append("--with-ragas")
        if args.ragas_limit:
            eval_cmd.extend(["--ragas-limit", str(args.ragas_limit)])
        if args.llm_model:
            eval_cmd.extend(["--llm-model", args.llm_model])

        subprocess.run(eval_cmd, check=True)
        results_by_pair[pair_key][parser_type] = get_latest_metrics(collection, run_name)

        # 3. Clean up collection
        if not args.keep_collections:
            try:
                get_qdrant_client(QDRANT_URL).delete_collection(collection)
                print(f"Коллекция '{collection}' очищена.")
            except Exception:
                pass

    # Обновление общего отчета
    subprocess.run([sys.executable, summary_script], check=True)

    # Печать красивой сравнительной таблицы
    print_comparison_table(results_by_pair)


if __name__ == "__main__":
    main()
