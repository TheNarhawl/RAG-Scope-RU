import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import EXPERIMENTS_DIR

OUT_PATH = EXPERIMENTS_DIR / "RESULTS.md"

# НАСТРОЙКА КОЛОНОК ТАБЛИЦЫ
COLUMNS_CONFIG = {
    "#": True,
    "Tag": True,
    "Embedding": True,
    "Chunk": True,
    "Overlap": True,
    "N": True,
    "Hit@1": True,
    "Hit@3": True,
    "Hit@5": True,
    "Hit@10": True,
    "MRR": True,
    "Faithful": True,
    "AnsCorr": True,
    "ms/q": True,
    "Timestamp": True,
}


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(EXPERIMENTS_DIR.glob("*/metrics_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Пропущен {path.name}: {e}")
            continue

        cfg = data.get("config", {})
        summary = data.get("summary", {})
        hit_rate = summary.get("hit_rate", {})
        ragas = summary.get("ragas", {})

        runs.append({
            "file": path.name,
            "collection": cfg.get("ingest", {}).get("collection", "—"),
            "embedding": cfg.get("ingest", {}).get("embedding_model", "—"),
            "chunk_size": cfg.get("ingest", {}).get("chunk_size", "—"),
            "overlap": cfg.get("ingest", {}).get("chunk_overlap", "—"),
            "tag": data.get("tag", "—"),
            "timestamp": data.get("timestamp", "—"),
            "n": summary.get("n_questions", "—"),
            "hit@1": hit_rate.get("@1"),
            "hit@3": hit_rate.get("@3"),
            "hit@5": hit_rate.get("@5"),
            "hit@10": hit_rate.get("@10"),
            "mrr": summary.get("mrr"),
            "ms_per_query": summary.get("ms_per_query"),
            "faithfulness": ragas.get("faithfulness"),
            "answer_correctness": ragas.get("answer_correctness"),
        })
    return runs


def fmt_float(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def fmt_ms(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.1f}"
    return str(x)


def fmt_any(x) -> str:
    if x is None:
        return "—"
    return str(x)


def short_model(name: str) -> str:
    """'sentence-transformers/all-MiniLM-L6-v2' -> 'all-MiniLM-L6-v2'"""
    if "/" in name:
        return name.split("/")[-1]
    return name


COLUMN_EXTRACTORS = [
    ("#", lambda r, i: str(i)),
    ("Tag", lambda r, i: fmt_any(r["tag"])),
    ("Embedding", lambda r, i: short_model(r["embedding"])),
    ("Chunk", lambda r, i: fmt_any(r["chunk_size"])),
    ("Overlap", lambda r, i: fmt_any(r["overlap"])),
    ("N", lambda r, i: fmt_any(r["n"])),
    ("Hit@1", lambda r, i: fmt_float(r["hit@1"])),
    ("Hit@3", lambda r, i: fmt_float(r["hit@3"])),
    ("Hit@5", lambda r, i: fmt_float(r["hit@5"])),
    ("Hit@10", lambda r, i: fmt_float(r["hit@10"])),
    ("MRR", lambda r, i: fmt_float(r["mrr"])),
    ("Faithful", lambda r, i: fmt_float(r["faithfulness"])),
    ("AnsCorr", lambda r, i: fmt_float(r["answer_correctness"])),
    ("ms/q", lambda r, i: fmt_ms(r["ms_per_query"])),
    ("Timestamp", lambda r, i: fmt_any(r["timestamp"])),
]


def build_markdown(runs: list[dict]) -> str:
    lines = []
    lines.append("# Experiment results\n")
    lines.append(f"_Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append(f"Всего прогонов: **{len(runs)}**\n")

    if not runs:
        lines.append("_Пока нет ни одного прогона._")
        return "\n".join(lines)

    active_columns = [
        (name, fn)
        for name, fn in COLUMN_EXTRACTORS
        if COLUMNS_CONFIG.get(name, True)
    ]

    if not active_columns:
        lines.append("_Все колонки таблицы отключены в COLUMNS_CONFIG._")
        return "\n".join(lines)

    headers = [name for name, _ in active_columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    runs_sorted = sorted(runs, key=lambda r: r["timestamp"])

    for i, r in enumerate(runs_sorted, start=1):
        cells = [fn(r, i) for _, fn in active_columns]
        assert len(cells) == len(headers), f"Строка {i}: {len(cells)} ячеек вместо {len(headers)}"
        lines.append("| " + " | ".join(cells) + " |")

    ragas_active = COLUMNS_CONFIG.get("Faithful", True) or COLUMNS_CONFIG.get("AnsCorr", True)
    has_ragas = any(r["faithfulness"] is not None for r in runs_sorted)
    if ragas_active and not has_ragas:
        lines.append(
            "\nFaithful и AnsCorr (RAGAS) появятся после подключения оценки "
            "генерации через LLM + RAGAS.\n"
        )

    valid = [r for r in runs_sorted if r["mrr"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["mrr"])
        lines.append(f"\n## Best run (по MRR)\n")
        lines.append(f"- **Tag:** `{best['tag']}`")
        lines.append(f"- **Embedding:** `{best['embedding']}`")
        lines.append(f"- **Chunk:** {best['chunk_size']} / overlap {best['overlap']}")
        lines.append(f"- **MRR:** {best['mrr']:.3f}")
        lines.append(f"- **Hit@1:** {fmt_float(best['hit@1'])}")
        lines.append(f"- **Файл:** `{best['file']}`\n")

    return "\n".join(lines)


def main():
    runs = load_runs()
    md = build_markdown(runs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"Сохранено: {OUT_PATH}")
    print(f"Прогонов в сводке: {len(runs)}")


if __name__ == "__main__":
    main()