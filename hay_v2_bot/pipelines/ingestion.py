"""Пайплайн разбора файла: Docling, метаданные, эмбеддинг OpenAI, Pinecone."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from loguru import logger

from components.converter import build_docling_converter
from components.memory import indexing_writer_policy
from components.metadata import DocumentMetadataEnricher
from components.openai_factory import document_embedder
from components.settings import Settings


def build_ingestion_pipeline(settings: Settings, document_store: PineconeDocumentStore) -> Pipeline:
    """Собирает загрузку файла в векторную базу.

    Docling конвертирует PDF, DOCX и другие поддерживаемые форматы
    и режет документ на чанки. Каждый чанк получает имя файла и номер.
    Эмбеддинг считает OpenAI ``text-embedding-3-small`` напрямую,
    затем ``DocumentWriter`` пишет вектор в Pinecone.

    Параметры:
        settings: Модель, ключ и прямой адрес OpenAI.
        document_store: Общий индекс Pinecone.

    Возвращает:
        Готовый ``Pipeline``. При запуске передайте ``converter.sources``
        и поля ``enricher`` с именем файла и идентификаторами пользователя.
    """
    logger.info("Собираю ingestion pipeline и конвертер Docling")
    pipeline = Pipeline()
    pipeline.add_component("converter", build_docling_converter())
    pipeline.add_component("enricher", DocumentMetadataEnricher())
    pipeline.add_component("embedder", document_embedder(settings))
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store, policy=indexing_writer_policy()),
    )
    pipeline.connect("converter.documents", "enricher.documents")
    pipeline.connect("enricher.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")
    logger.info(
        "Собран ingestion pipeline: DoclingConverter -> DocumentMetadataEnricher "
        "-> OpenAIDocumentEmbedder -> DocumentWriter"
    )
    return pipeline
