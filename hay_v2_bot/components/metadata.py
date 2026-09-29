"""Плоские метаданные чанков перед записью в Pinecone.

Pinecone принимает в метаданных только строки, числа, логические значения
и списки строк. Вложенный ``dl_meta`` Docling сюда не кладётся: номер
страницы и заголовки из него переносятся в отдельные поля.
"""

from __future__ import annotations

import hashlib

from haystack import Document, component
from loguru import logger

from components.settings import DOCUMENT_SOURCE

try:
    from docling.chunking import DocChunk
except ImportError:  # pragma: no cover - зависит от версии docling
    DocChunk = None  # type: ignore[misc, assignment]


def _page_and_headings(meta: dict) -> tuple[int | None, str | None]:
    """Достаёт номер страницы и заголовки из метаданных чанка Docling.

    Сначала пробует модель ``DocChunk``. Если формат другой, обходит словарь
    и ищет ``page_no`` в данных о положении фрагмента на странице.
    """
    dl_meta = meta.get("dl_meta")
    if dl_meta is None:
        return None, None

    if DocChunk is not None:
        try:
            chunk = DocChunk.model_validate(dl_meta)
            page = None
            items = list(chunk.meta.doc_items or [])
            if items and items[0].prov:
                page = int(items[0].prov[0].page_no)
            headings = None
            if chunk.meta.headings:
                headings = " / ".join(str(item) for item in chunk.meta.headings if item)
            return page, headings
        except (TypeError, ValueError, AttributeError) as error:
            logger.debug("dl_meta не разобрался моделью DocChunk: {}", error)

    payload = dl_meta if isinstance(dl_meta, dict) else {}
    inner = payload.get("meta") if isinstance(payload.get("meta"), dict) else payload
    if not isinstance(inner, dict):
        return None, None

    page = None
    for item in inner.get("doc_items") or []:
        if not isinstance(item, dict):
            continue
        for entry in item.get("prov") or []:
            if isinstance(entry, dict) and entry.get("page_no") is not None:
                page = int(entry["page_no"])
                break
        if page is not None:
            break

    raw_headings = inner.get("headings")
    headings = None
    if isinstance(raw_headings, list):
        parts = [str(item) for item in raw_headings if item]
        if parts:
            headings = " / ".join(parts)
    return page, headings


def _document_id(user_id: str, file_name: str, chunk_index: int) -> str:
    """Стабильный идентификатор чанка: повторная загрузка того же файла перезаписывает его."""
    digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:12]
    return f"file-{user_id}-{digest}-{chunk_index}"


@component
class DocumentMetadataEnricher:
    """Дописывает имя файла, номер чанка и страницу и убирает вложенные метаданные.

    Компонент стоит в ingestion-пайплайне между Docling и эмбеддером.
    Пустые чанки отбрасываются и в Pinecone не попадают.
    """

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        file_name: str,
        user_id: str,
        chat_id: str,
        username: str | None = None,
        full_name: str | None = None,
    ) -> dict[str, list[Document]]:
        """Собирает плоские метаданные для каждого непустого чанка.

        Параметры:
            documents: Чанки, которые вернул DoclingConverter.
            file_name: Имя файла, как его прислал пользователь.
            user_id: Идентификатор пользователя Telegram строкой.
            chat_id: Идентификатор чата строкой.
            username: Username без «собаки», если он есть в профиле.
            full_name: Имя и фамилия из профиля Telegram.

        Возвращает:
            Словарь с ключом ``documents``: список документов Haystack
            с полями ``file_name``, ``chunk_index`` и, если страница известна,
            ``page_number``.
        """
        enriched: list[Document] = []
        for document in documents:
            content = (document.content or "").strip()
            if not content:
                logger.info("Пустой чанк файла {} пропущен", file_name)
                continue
            chunk_index = len(enriched) + 1
            page, headings = _page_and_headings(document.meta)
            meta: dict[str, str | int] = {
                "file_name": file_name[:200],
                "chunk_index": chunk_index,
                "source": DOCUMENT_SOURCE,
                "user_id": user_id,
                "chat_id": chat_id,
            }
            if page is not None:
                meta["page_number"] = page
            if headings:
                meta["headings"] = headings[:300]
            if username:
                meta["username"] = username
            if full_name:
                meta["full_name"] = full_name[:200]
            enriched.append(
                Document(
                    id=_document_id(user_id, file_name, chunk_index),
                    content=content,
                    meta=meta,
                )
            )
            logger.info(
                "Чанк {} файла {}: page={}, символов={}, id={}",
                chunk_index,
                file_name,
                page,
                len(content),
                enriched[-1].id,
            )
        logger.info("Файл {}: к записи подготовлено {} чанков", file_name, len(enriched))
        return {"documents": enriched}
