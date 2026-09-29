"""Нарезка Docling-документа токенайзером text-embedding-3-small.

Токенайзер только считает токены и режет чанки. Вектор по этим чанкам
считает удалённый OpenAI, а не локальная модель.
"""

from __future__ import annotations

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from loguru import logger

from components.settings import CHUNK_MAX_TOKENS, EMBEDDING_MODEL


class SafeOpenAITokenizer(OpenAITokenizer):
    """Токенайзер OpenAI, который не падает на служебных кусках текста.

    Обычный ``tiktoken.encode`` поднимает ошибку, если в документе
    встретилась последовательность вроде служебного токена. Для нарезки
    файла такие куски нужно просто посчитать, а не отклонять документ.
    """

    def count_tokens(self, text: str) -> int:
        """Возвращает число токенов ``text-embedding-3-small`` в строке."""
        return len(self.tokenizer.encode(text, disallowed_special=()))


def build_chunker() -> HybridChunker:
    """Собирает HybridChunker под токенизацию ``text-embedding-3-small``.

    ``max_tokens`` здесь — длина одного чанка для поиска, а не предел модели.
    Модель принимает до 8191 токенов, в индекс пишутся более короткие куски,
    чтобы ретривер находил точный фрагмент.

    Возвращает:
        Чанкер Docling, который отдаёт страницы и разделы отдельными кусками.
    """
    try:
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    except KeyError:
        logger.warning("У tiktoken нет кодировки {}. Беру cl100k_base.", EMBEDDING_MODEL)
        encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = SafeOpenAITokenizer(tokenizer=encoding, max_tokens=CHUNK_MAX_TOKENS)
    logger.info(
        "Чанкер Docling: токены {} через tiktoken, не больше {} токенов на чанк",
        EMBEDDING_MODEL,
        CHUNK_MAX_TOKENS,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)
