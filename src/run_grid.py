import subprocess
import sys
from pathlib import Path

from paths import CONFIGS_DIR


def main():
    # Находим все папки чанков (где лежит dense.yaml)
    chunk_dirs = sorted(p.parent for p in CONFIGS_DIR.rglob("dense.yaml") if p.parent != CONFIGS_DIR)

    for c_dir in chunk_dirs:
        rel_dir = c_dir.relative_to(CONFIGS_DIR).as_posix()
        print(f"\n==================== ГРУППА: {rel_dir} ====================")

        # 1. Индексация (один раз на группу чанка)
        subprocess.run([sys.executable, "src/ingest.py", "--config", f"{rel_dir}/dense"], check=True)

        # 2. Оценка всех стратегий в этой папке
        for strat in ["dense", "hybrid", "hybrid-rerank"]:
            cfg_File = c_dir / f"{strat}.yaml"
            if cfg_File.exists():
                subprocess.run([sys.executable, "src/evaluate_retrieval.py", "--config", f"{rel_dir}/{strat}"], check=True)

    # 3. Сводка результатов
    subprocess.run([sys.executable, "src/summarize_experiments.py"], check=True)


if __name__ == "__main__":
    main()