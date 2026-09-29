"""Пайплайны записи и поиска долговременной памяти диалога."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from loguru import logger

from components.memory import indexing_writer_policy
from components.openai_factory import document_embedder, text_embedder
from components.settings import Settings


def build_memory_index_pipeline(settings: Settings, document_store: PineconeDocumentStore) -> Pipeline:
    """Пайплайн записи реплики: эмбеддинг OpenAI и upsert в Pinecone.

    Эмбеддер ходит на прямой ``/embeddings`` OpenAI моделью ``text-embedding-3-small``.
    """
    pipeline = Pipeline()
    pipeline.add_component("embedder", document_embedder(settings))
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store, policy=indexing_writer_policy()),
    )
    pipeline.connect("embedder.documents", "writer.documents")
    logger.info("Собран memory index pipeline: OpenAIDocumentEmbedder -> DocumentWriter")
    return pipeline


def build_memory_query_pipeline(settings: Settings, document_store: PineconeDocumentStore) -> Pipeline:
    """Пайплайн поиска ближайшей реплики памяти перед записью дубликата."""
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder(settings))
    pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=document_store))
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    logger.info("Собран memory query pipeline: OpenAITextEmbedder -> PineconeEmbeddingRetriever")
    return pipeline
