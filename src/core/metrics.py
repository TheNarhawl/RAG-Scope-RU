def slugify(s: str) -> str:
    """Преобразует строку в безопасный идентификатор/имя файла."""
    return s.lower().replace(" ", "_").replace("+", "").replace("-", "_")


def hit_at_k(sources: list[str], target: str, k: int) -> float:
    """Проверяет попадание целевого документа в топ-k результатов."""
    return 1.0 if target in sources[:k] else 0.0


def rr(sources: list[str], target: str) -> float:
    """Вычисляет Reciprocal Rank (1 / позиция целевого документа)."""
    for i, s in enumerate(sources, start=1):
        if s == target:
            return 1.0 / i
    return 0.0
