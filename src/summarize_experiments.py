import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import EXPERIMENTS_DIR

OUT_PATH = EXPERIMENTS_DIR / "RESULTS.md"


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


def build_markdown(runs: list[dict]) -> str:
    lines = []
    lines.append("# Experiment results\n")
    lines.append(f"_Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append(f"Всего прогонов: **{len(runs)}**\n")

    if not runs:
        lines.append("_Пока нет ни одного прогона._")
        return "\n".join(lines)

    headers = [
        "#", "Tag", "Embedding", "Chunk", "Overlap", "N",
        "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR",
        "Faithful", "AnsCorr",
        "ms/q", "Timestamp",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    runs_sorted = sorted(runs, key=lambda r: r["timestamp"])

    for i, r in enumerate(runs_sorted, start=1):
        cells = [
            str(i),
            fmt_any(r["tag"]),
            short_model(r["embedding"]),
            fmt_any(r["chunk_size"]),
            fmt_any(r["overlap"]),
            fmt_any(r["n"]),
            fmt_float(r["hit@1"]),
            fmt_float(r["hit@3"]),
            fmt_float(r["hit@5"]),
            fmt_float(r["hit@10"]),
            fmt_float(r["mrr"]),
            fmt_float(r["faithfulness"]),
            fmt_float(r["answer_correctness"]),
            fmt_ms(r["ms_per_query"]),
            fmt_any(r["timestamp"]),
        ]
        assert len(cells) == len(headers), f"Строка {i}: {len(cells)} ячеек вместо {len(headers)}"
        lines.append("| " + " | ".join(cells) + " |")

    # Пояснение к пустым колонкам
    has_ragas = any(r["faithfulness"] is not None for r in runs_sorted)
    if not has_ragas:
        lines.append(
            "\nFaithful и AnsCorr (RAGAS) появятся после подключения оценки "
            "генерации через LLM + RAGAS.\n"
        )

    # Best run по MRR
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