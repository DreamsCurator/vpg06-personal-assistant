"""Общее хранилище Pinecone для памяти диалога и чанков файлов.

Оба типа записей лежат в одном индексе и различаются полем ``source``.
Метрика индекса — cosine, размерность совпадает с ``text-embedding-3-small``.
"""

from __future__ import annotations

from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from loguru import logger

from components.settings import Settings


def build_document_store(settings: Settings) -> PineconeDocumentStore:
    """Открывает индекс Pinecone или создаёт serverless-индекс, если его нет.

    ``dimension`` нужен только при создании. Уже существующий индекс
    должен быть собран с той же размерностью, что и модель эмбеддингов.

    Параметры:
        settings: Ключ, имя индекса, пространство имён и размерность.

    Возвращает:
        Подключённый ``PineconeDocumentStore``.
    """
    logger.info(
        "Подключаю Pinecone: index={}, namespace={}, metric=cosine, dimension={}",
        settings.pinecone_index,
        settings.pinecone_namespace,
        settings.embedding_dimension,
    )
    return PineconeDocumentStore(
        api_key=Secret.from_token(settings.pinecone_api_key),
        index=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        metric="cosine",
        dimension=settings.embedding_dimension,
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
        show_progress=False,
    )
