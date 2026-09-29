"""Фабрики клиентов Haystack для удалённого OpenAI.

Все генераторы и эмбеддеры ходят напрямую на ``https://api.openai.com/v1``.
Локальные классы вроде SentenceTransformers здесь не создаются.
"""

from __future__ import annotations

from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from components.settings import Settings


def _embedder_kwargs(settings: Settings) -> dict[str, object]:
    """Общие аргументы эмбеддеров OpenAI: модель small, размерность и прямой API."""
    kwargs: dict[str, object] = {
        "api_key": Secret.from_token(settings.openai_api_key),
        "model": settings.embedding_model,
        "api_base_url": settings.openai_base_url,
    }
    if settings.embedding_model.startswith("text-embedding-3"):
        kwargs["dimensions"] = settings.embedding_dimension
    return kwargs


def document_embedder(settings: Settings) -> OpenAIDocumentEmbedder:
    """Эмбеддер документов: тексты чанков уходят на ``/embeddings`` OpenAI.

    Модель берётся из настроек. По умолчанию это ``text-embedding-3-small``.
    """
    return OpenAIDocumentEmbedder(**_embedder_kwargs(settings), progress_bar=False)


def text_embedder(settings: Settings) -> OpenAITextEmbedder:
    """Эмбеддер поискового запроса той же удалённой моделью, что и документы."""
    return OpenAITextEmbedder(**_embedder_kwargs(settings))


def chat_generator(settings: Settings) -> OpenAIChatGenerator:
    """Чат-модель OpenAI через тот же прямой адрес, что и эмбеддинги."""
    return OpenAIChatGenerator(
        api_key=Secret.from_token(settings.openai_api_key),
        model=settings.chat_model,
        api_base_url=settings.openai_base_url,
    )
