"""Долговременная память диалога: поиск похожей реплики и запись в Pinecone."""

from __future__ import annotations

import threading
import uuid

from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy
from loguru import logger

from components.settings import COSINE_SIMILARITY_THRESHOLD, DOCUMENT_SOURCE, MEMORY_SOURCE


def memory_filter(user_id: int, role: str | None = None) -> dict:
    """Фильтр Haystack: только память этого пользователя, при необходимости одной роли.

    Документы файлов в выборку не попадают: у них другой ``source``.
    """
    conditions: list[dict[str, str]] = [
        {"field": "meta.user_id", "operator": "==", "value": str(user_id)},
        {"field": "meta.source", "operator": "==", "value": MEMORY_SOURCE},
    ]
    if role is not None:
        conditions.append({"field": "meta.role", "operator": "==", "value": role})
    return {"operator": "AND", "conditions": conditions}


def document_filter(user_id: int) -> dict:
    """Фильтр Haystack: только чанки файлов этого пользователя."""
    return {
        "operator": "AND",
        "conditions": [
            {"field": "meta.user_id", "operator": "==", "value": str(user_id)},
            {"field": "meta.source", "operator": "==", "value": DOCUMENT_SOURCE},
        ],
    }


def retrieve_memories(
    query_pipeline: Pipeline,
    pipeline_lock: threading.Lock,
    text: str,
    user_id: int,
    top_k: int,
    role: str | None = None,
) -> list[Document]:
    """Ищет фрагменты памяти, ближайшие к тексту по косинусному сходству."""
    logger.info("Ищу память user_id={} role={} top_k={}", user_id, role, top_k)
    with pipeline_lock:
        result = query_pipeline.run(
            {
                "text_embedder": {"text": text},
                "retriever": {
                    "filters": memory_filter(user_id, role),
                    "top_k": top_k,
                },
            }
        )
    documents = list(result["retriever"]["documents"])
    logger.info("Память user_id={}: найдено {}", user_id, len(documents))
    return documents


def remember(
    index_pipeline: Pipeline,
    query_pipeline: Pipeline,
    pipeline_lock: threading.Lock,
    text: str,
    user_id: int,
    chat_id: int,
    role: str,
    username: str | None,
    full_name: str | None,
) -> None:
    """Пишет реплику в Pinecone, если это не повтор уже сохранённого фрагмента.

    Сначала ищется ближайший фрагмент той же роли. Сходство берётся из score
    Pinecone при метрике cosine: чем выше, тем ближе по смыслу.
    Ниже ``COSINE_SIMILARITY_THRESHOLD`` создаётся новый слот.
    На границе и выше тот же текст пропускается, другая формулировка
    перезаписывает найденный слот.
    """
    if not text.strip():
        return

    matches = retrieve_memories(query_pipeline, pipeline_lock, text, user_id, top_k=1, role=role)
    document_id = str(uuid.uuid4())
    similarity = None
    action = "remembered"
    if matches:
        nearest = matches[0]
        similarity = nearest.score
        same_text = (nearest.content or "").strip() == text.strip()
        if similarity is not None and similarity >= COSINE_SIMILARITY_THRESHOLD:
            if same_text:
                logger.info(
                    "Память user_id={} role={} action=skipped similarity={:.3f}",
                    user_id,
                    role,
                    similarity,
                )
                return
            document_id = nearest.id
            action = "updated"

    meta = {
        "user_id": str(user_id),
        "chat_id": str(chat_id),
        "role": role,
        "source": MEMORY_SOURCE,
    }
    if username:
        meta["username"] = username
    if full_name:
        meta["full_name"] = full_name

    document = Document(id=document_id, content=text.strip(), meta=meta)
    with pipeline_lock:
        index_pipeline.run({"embedder": {"documents": [document]}})
    logger.info(
        "Память user_id={} role={} action={} similarity={}",
        user_id,
        role,
        action,
        None if similarity is None else round(similarity, 3),
    )


def indexing_writer_policy() -> DuplicatePolicy:
    """Политика записи: повторный идентификатор заменяет вектор в Pinecone."""
    return DuplicatePolicy.OVERWRITE
