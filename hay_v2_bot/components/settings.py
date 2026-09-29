"""Настройки бота из окружения и постоянные параметры пайплайнов.

Эмбеддинги считаются только удалённой моделью OpenAI ``text-embedding-3-small``.
Чат и эмбеддинги всегда идут на ``https://api.openai.com/v1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Корень репозитория: hay_v2_bot/components/settings.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Прямой API OpenAI. Адрес из окружения не читается, чтобы запрос не ушёл на прокси.
OPENAI_BASE_URL = "https://api.openai.com/v1"

CHAT_MODEL_DEFAULT = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Граница косинусного сходства для долговременной памяти диалога.
COSINE_SIMILARITY_THRESHOLD = 0.85
MEMORY_TOP_K = 6
DOCUMENT_TOP_K = 5
HISTORY_LIMIT = 8
# Размер чанка в токенах text-embedding-3-small. Сама модель принимает до 8191,
# но для поиска удобнее короткие фрагменты. Это не локальный эмбеддинг:
# токенайзер только режет текст, вектор считает OpenAI.
CHUNK_MAX_TOKENS = 512
SUMMARY_CHAR_LIMIT = 12000

MEMORY_SOURCE = "haystack-assistant"
DOCUMENT_SOURCE = "document"

CAT_FACT_URL = "https://catfact.ninja/fact"
DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"

FILE_RECEIVED_TEXT = "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
FILE_READY_TEXT = "Готово. Я изучил этот файл, теперь можем его обсудить."

SUPPORTED_SUFFIXES = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".md",
        ".txt",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }
)

MIME_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}

# Лимит Telegram Bot API на скачивание файла через getFile.
TELEGRAM_FILE_LIMIT_BYTES = 20 * 1024 * 1024

_REQUIRED_ENV = ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "TELEGRAM_BOT_TOKEN")


@dataclass(frozen=True)
class Settings:
    """Снимок настроек, с которым собираются клиенты и пайплайны.

    Атрибуты:
        openai_api_key: Ключ OpenAI.
        openai_base_url: Прямой адрес API OpenAI, без завершающего слэша.
        chat_model: Модель диалога и краткого резюме.
        embedding_model: Удалённая модель эмбеддингов, по умолчанию small.
        embedding_dimension: Размерность вектора этой модели.
        pinecone_api_key: Ключ Pinecone.
        pinecone_index: Имя уже существующего или создаваемого индекса.
        pinecone_namespace: Пространство имён индекса.
        telegram_token: Токен Telegram-бота.
    """

    openai_api_key: str = field(repr=False)
    openai_base_url: str
    chat_model: str
    embedding_model: str
    embedding_dimension: int
    pinecone_api_key: str = field(repr=False)
    pinecone_index: str
    pinecone_namespace: str
    telegram_token: str = field(repr=False)


def load_settings() -> Settings:
    """Читает ``.env`` в корне проекта и проверяет обязательные ключи.

    Если ``OPENAI_EMBEDDING_MODEL`` указывает не на модель OpenAI,
    подставляется ``text-embedding-3-small``: локально эмбеддинги не считаются.
    Адрес API всегда ``https://api.openai.com/v1``, переменная ``OPENAI_BASE_URL``
    не используется.

    Возвращает:
        Заполненный ``Settings``.

    Вызывает:
        ValueError: Если не задан ключ OpenAI, Pinecone, индекс или токен Telegram.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise ValueError("Не заданы переменные окружения: " + ", ".join(missing) + ".")

    requested_embedding = (os.getenv("OPENAI_EMBEDDING_MODEL") or EMBEDDING_MODEL).strip()
    if requested_embedding not in EMBEDDING_DIMENSIONS:
        logger.warning(
            "Модель эмбеддингов {} не является удалённой моделью OpenAI. "
            "Использую {} и отправляю тексты напрямую в OpenAI.",
            requested_embedding,
            EMBEDDING_MODEL,
        )
        embedding_model = EMBEDDING_MODEL
    else:
        embedding_model = requested_embedding

    base_url = OPENAI_BASE_URL

    dimension = int(os.getenv("OPENAI_EMBEDDING_DIMENSION") or EMBEDDING_DIMENSIONS[embedding_model])
    settings = Settings(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_base_url=base_url,
        chat_model=(os.getenv("OPENAI_CHAT_MODEL") or CHAT_MODEL_DEFAULT).strip(),
        embedding_model=embedding_model,
        embedding_dimension=dimension,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        pinecone_index=os.environ["PINECONE_INDEX_NAME"],
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE") or "default",
        telegram_token=os.environ["TELEGRAM_BOT_TOKEN"],
    )
    logger.info(
        "Настройки: чат={}, эмбеддинги={} ({} измерений), endpoint={}, индекс={}, namespace={}",
        settings.chat_model,
        settings.embedding_model,
        settings.embedding_dimension,
        settings.openai_base_url,
        settings.pinecone_index,
        settings.pinecone_namespace,
    )
    logger.info("Локальная модель эмбеддингов не используется: векторы считает удалённый OpenAI")
    return settings
