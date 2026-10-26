from typing import List


def naive_chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Наивный посимвольный чанкинг (Baseline).
    """
    chunks, start = [], 0
    while start < len(text):
        c = text[start : start + size].strip()
        if c:
            chunks.append(c)
        start += size - overlap
    return chunks


def structural_chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: List[str] | None = None,
) -> List[str]:
    if separators is None:
        separators = [
            "\nРаздел ",
            "\nГлава ",
            "\nСтатья ",
            "\nПункт ",
            "\nПриложение ",
            "\n\n",
            "\n",
            ". ",
            "; ",
            " ",
            "",
        ]

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Ищем разделитель наивысшего приоритета, присутствующий в тексте
    chosen_sep = separators[-1]
    next_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            chosen_sep = ""
            break
        if sep in text:
            chosen_sep = sep
            next_separators = separators[i + 1 :]
            break

    # Разбиваем по выбранному разделителю с сохранением контекстных маркеров
    if chosen_sep != "":
        if chosen_sep.startswith("\n"):
            # Структурный префикс (Статья, Глава): сохраняем в начале блока
            raw_parts = text.split(chosen_sep)
            parts = [raw_parts[0]] + [
                chosen_sep.lstrip("\n") + p for p in raw_parts[1:]
            ]
            parts = [p.strip() for p in parts if p.strip()]
        elif chosen_sep in [". ", "; "]:
            # Предложения: сохраняем знак препинания в конце
            raw_parts = text.split(chosen_sep)
            punct = chosen_sep.strip()
            parts = [p.strip() + punct for p in raw_parts if p.strip()]
        else:
            parts = [p.strip() for p in text.split(chosen_sep) if p.strip()]
    else:
        # Крайний случай (если нет пробелов) — посимвольный срез
        parts = list(text)

    chunks = []
    current_parts = []
    current_len = 0

    for part in parts:
        part_len = len(part)

        # Если один блок сам по себе превышает chunk_size — дробим его следующим разделителем
        if part_len > chunk_size:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_len = 0
            sub_chunks = structural_chunk_text(
                part, chunk_size, chunk_overlap, next_separators
            )
            chunks.extend(sub_chunks)
            continue

        joiner_len = 1 if current_parts else 0
        if current_len + joiner_len + part_len > chunk_size:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())

                # Расчет overlap: берем последние смысловые части, укладывающиеся в chunk_overlap
                tail = []
                tail_len = 0
                for p in reversed(current_parts):
                    p_len = len(p) + (1 if tail else 0)
                    if tail_len + p_len <= chunk_overlap:
                        tail.insert(0, p)
                        tail_len += p_len
                    else:
                        break
                current_parts = tail
                current_len = tail_len

        joiner_len = 1 if current_parts else 0
        current_parts.append(part)
        current_len += joiner_len + part_len

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return [c for c in chunks if c]