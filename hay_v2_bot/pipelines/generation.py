"""Пайплайн ответа: эмбеддинг запроса, два ретривера и агент."""

from __future__ import annotations

from haystack import Pipeline
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from loguru import logger

from components.openai_factory import text_embedder
from components.responder import AssistantResponder
from components.settings import DOCUMENT_TOP_K, MEMORY_TOP_K, Settings


def build_generation_pipeline(settings: Settings, document_store: PineconeDocumentStore) -> Pipeline:
    """Собирает ответ пользователю с учётом файлов и памяти диалога.

    Один ``OpenAITextEmbedder`` считает вектор запроса напрямую в OpenAI.
    Этот вектор получают два ретривера Pinecone: чанки документов
    и реплики долговременной памяти. ``AssistantResponder`` отдаёт оба
    набора модели и при необходимости вызывает инструменты.

    Фильтры по пользователю передаются при запуске, чтобы чужие файлы
    и чужая память не попадали в ответ.
    """
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder(settings))
    pipeline.add_component(
        "document_retriever",
        PineconeEmbeddingRetriever(document_store=document_store, top_k=DOCUMENT_TOP_K),
    )
    pipeline.add_component(
        "memory_retriever",
        PineconeEmbeddingRetriever(document_store=document_store, top_k=MEMORY_TOP_K),
    )
    pipeline.add_component("assistant", AssistantResponder(settings))
    pipeline.connect("text_embedder.embedding", "document_retriever.query_embedding")
    pipeline.connect("text_embedder.embedding", "memory_retriever.query_embedding")
    pipeline.connect("document_retriever.documents", "assistant.documents")
    pipeline.connect("memory_retriever.documents", "assistant.memories")
    logger.info(
        "Собран generation pipeline: OpenAITextEmbedder -> "
        "document retriever (top_k={}) и memory retriever (top_k={}) -> AssistantResponder",
        DOCUMENT_TOP_K,
        MEMORY_TOP_K,
    )
    return pipeline
