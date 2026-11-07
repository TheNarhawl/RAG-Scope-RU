"""Устаревший шлюз для обратной совместимости. Основной скрипт перемещен в scripts/make_configs.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "make_configs.py"
    runpy.run_path(str(target), run_name="__main__")
