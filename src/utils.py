import os
import random
import numpy as np
import torch
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer


def set_seed(seed: int = 42) -> None:
    """Фиксирует генераторы случайных чисел для воспроизводимости экспериментов."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(verbose: bool = True) -> str:
    """Определяет доступное устройство (CUDA / CPU)."""
    if torch.cuda.is_available():
        if verbose:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Device: cuda ({name}, {vram:.1f} GB VRAM)")
        return "cuda"
    if verbose:
        print("Device: cpu")
    return "cpu"


def get_qdrant_client(url: str) -> QdrantClient:
    """Создает клиент Qdrant без назойливых предупреждений о разнице минорных версий."""
    return QdrantClient(url=url, check_compatibility=False)


def _fix_non_persistent_buffers(st_model: SentenceTransformer, device: str) -> None:
    """
    Исправляет баг transformers >= 5.0 (Issue #43950), при котором буферы с persistent=False
    (position_ids, RoPE cos/sin кэши в Alibaba-NLP/gte-multilingual-base и др.)
    оказываются заполнены мусором после материализации с meta-device.
    """
    try:
        first_mod = st_model._first_module()
        auto_model = getattr(first_mod, "auto_model", None)
        if auto_model is None:
            return

        embeddings = getattr(auto_model, "embeddings", None)
        if embeddings is None:
            return

        # 1. Восстанавливаем корректный тензор position_ids (0, 1, ..., max_pos - 1)
        if hasattr(embeddings, "position_ids") and embeddings.position_ids is not None:
            pos = embeddings.position_ids
            if pos.ndim == 1:
                clean_pos = torch.arange(pos.size(0), device=pos.device, dtype=torch.long)
            else:
                clean_pos = torch.arange(pos.size(-1), device=pos.device, dtype=torch.long).unsqueeze(0)
            embeddings.register_buffer("position_ids", clean_pos, persistent=False)

        # 2. Переинициализируем RoPE-буферы (inv_freq, cos_cached, sin_cached) для GTE / кастомных моделей
        if hasattr(embeddings, "_init_rope") and hasattr(auto_model, "config"):
            embeddings._init_rope(auto_model.config)
            embeddings.to(device)
    except Exception as e:
        print(f"[WARN] Не удалось проверить буферы модели: {e}")


def load_dense_model(model_name: str, device: str | None = None, max_seq_length: int = 1024) -> SentenceTransformer:
    """Загружает SentenceTransformer с поддержкой кастомного кода и защитой буферов."""
    if device is None:
        device = pick_device(verbose=False)

    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
    )
    _fix_non_persistent_buffers(model, device)

    # Ограничиваем максимальную длину последовательности в разумных пределах под чанки до 2048 символов
    if model.max_seq_length is None or model.max_seq_length > max_seq_length:
        model.max_seq_length = max_seq_length

    return model


def load_reranker_model(model_name: str, device: str | None = None) -> CrossEncoder:
    """Загружает CrossEncoder на целевое устройство."""
    if device is None:
        device = pick_device(verbose=False)
    return CrossEncoder(
        model_name,
        device=device,
        trust_remote_code=True,
    )