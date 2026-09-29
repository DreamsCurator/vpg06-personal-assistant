"""Конвертер Docling для ingestion-пайплайна.

PDF читается через pypdfium2. Встроенный backend docling-parse на этой
установке ищет таблицу глифов по пути ``glyphs//standard`` и не открывает
ни один PDF. DOCX, PPTX и остальные форматы идут своими backend-ами Docling.
"""

from __future__ import annotations

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from haystack_integrations.components.converters.docling import DoclingConverter, ExportType
from loguru import logger

from components.chunking import build_chunker


def build_docling_converter() -> DoclingConverter:
    """Собирает DoclingConverter с чанками и рабочим PDF-backend.

    Режим ``DOC_CHUNKS`` отдаёт каждый фрагмент отдельным документом Haystack.
    Чанкер считает токены ``text-embedding-3-small`` и не считает эмбеддинг.

    Возвращает:
        Компонент Haystack. На вход ``sources`` принимает локальные пути.
    """
    document_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend),
        }
    )
    logger.info("Docling: PDF через pypdfium2, остальные форматы — штатные backend-ы")
    return DoclingConverter(
        converter=document_converter,
        export_type=ExportType.DOC_CHUNKS,
        chunker=build_chunker(),
    )
