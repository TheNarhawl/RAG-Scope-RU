import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_PATH = EXP_DIR / "RESULTS.md"

def load_runs() -> list[dict]:
    runs = []
    for path in sorted(EXP_DIR.glob("*/metrics_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Пропущен {path.name}: {e}")
            continue

        cfg = data.get("config", {})
        runs.append({
            "file": path.name,
            "collection": cfg.get("ingest", {}).get("collection", "—"),
            "embedding": cfg.get("ingest", {}).get("embedding_model", "—"),
            "chunk_size": cfg.get("ingest", {}).get("chunk_size", "—"),
            "overlap": cfg.get("ingest", {}).get("chunk_overlap", "—"),
            "tag": data.get("tag", "—"),
            "timestamp": data.get("timestamp", "—"),
            "n": data["summary"]["n_questions"],
            "hit@1": data["summary"]["hit_rate"]["@1"],
            "hit@3": data["summary"]["hit_rate"]["@3"],
            "hit@5": data["summary"]["hit_rate"]["@5"],
            "hit@10": data["summary"]["hit_rate"].get("@10", None),
            "ms_per_query": data["summary"].get("ms_per_query", None),
            "mrr": data["summary"]["mrr"],
        })
    return runs


def fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
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

    headers = ["#", "Tag", "Embedding", "Chunk", "Overlap", "N", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR", "ms/q", "Timestamp"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    runs_sorted = sorted(runs, key=lambda r: r["timestamp"])

    for i, r in enumerate(runs_sorted, start=1):
        lines.append("| " + " | ".join([
            str(i),
            r["tag"],
            short_model(r["embedding"]),
            str(r["chunk_size"]),
            str(r["overlap"]),
            str(r["n"]),
            fmt(r["hit@1"]),
            fmt(r["hit@3"]),
            fmt(r["hit@5"]),
            fmt(r["mrr"]),
            r["timestamp"],
        ]) + " |")

    best = max(runs_sorted, key=lambda r: r["mrr"])
    lines.append(f"\n## Best run (по MRR)\n")
    lines.append(f"- **Tag:** `{best['tag']}`")
    lines.append(f"- **Embedding:** `{best['embedding']}`")
    lines.append(f"- **Chunk:** {best['chunk_size']} / overlap {best['overlap']}")
    lines.append(f"- **MRR:** {best['mrr']:.3f}")
    lines.append(f"- **Hit@1:** {best['hit@1']:.3f}")
    lines.append(f"- **Файл:** `{best['file']}`\n")

    return "\n".join(lines)


def main():
    runs = load_runs()
    md = build_markdown(runs)
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"Сохранено: {OUT_PATH}")
    print(f"Прогонов в сводке: {len(runs)}")


if __name__ == "__main__":
    main()