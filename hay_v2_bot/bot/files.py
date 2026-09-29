"""Скачивание документа Telegram во временный файл для Docling."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import telebot
from loguru import logger

from components.settings import MIME_TO_SUFFIX, SUPPORTED_SUFFIXES, TELEGRAM_FILE_LIMIT_BYTES


class FileRejected(Exception):
    """Файл нельзя передать в Docling: формат, размер или пустое вложение."""


def supported_formats_text() -> str:
    """Человеческий список расширений для сообщения об ошибке."""
    return "PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, CSV, PNG, JPG, TIFF, BMP, WEBP"


def _safe_filename(name: str, mime_type: str | None) -> str:
    """Оставляет безопасное имя и дописывает расширение по MIME, если его не было."""
    cleaned = Path(name).name
    cleaned = "".join(char if char.isalnum() or char in "._- ()" else "_" for char in cleaned)
    cleaned = cleaned.strip(" .") or "document"
    suffix = Path(cleaned).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES and mime_type in MIME_TO_SUFFIX:
        cleaned = f"{Path(cleaned).stem}{MIME_TO_SUFFIX[mime_type]}"
    return cleaned


def rejection_reason(message: telebot.types.Message) -> str | None:
    """Возвращает текст отказа, если вложение нельзя отдать в Docling.

    ``None`` означает, что формат и размер подходят и можно писать
    пользователю, что анализ запущен.
    """
    document = message.document
    if document is None:
        return "В сообщении нет файла."
    if document.file_size and document.file_size > TELEGRAM_FILE_LIMIT_BYTES:
        return "Файл больше 20 МБ. Telegram не отдаёт боту такие вложения."
    filename = _safe_filename(document.file_name or "document", document.mime_type)
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        return "Этот формат я не разбираю. Пришлите файл одного из типов: " + supported_formats_text() + "."
    return None


def download_telegram_file(bot: telebot.TeleBot, message: telebot.types.Message) -> Path:
    """Скачивает вложение во временный каталог и возвращает путь к файлу.

    Docling выбирает разборщик по расширению, поэтому имя сохраняет
    исходный суффикс. Каталог нужно удалить после пайплайна через
    :func:`remove_download`.

    Вызывает:
        FileRejected: Если вложения нет, формат не поддержан или файл
            больше лимита скачивания Telegram.
    """
    document = message.document
    if document is None:
        raise FileRejected("В сообщении нет файла.")
    if document.file_size and document.file_size > TELEGRAM_FILE_LIMIT_BYTES:
        raise FileRejected("Файл больше 20 МБ. Telegram не отдаёт боту такие вложения.")

    filename = _safe_filename(document.file_name or "document", document.mime_type)
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise FileRejected(
            "Этот формат я не разбираю. Пришлите файл одного из типов: "
            + supported_formats_text()
            + "."
        )

    logger.info(
        "Скачиваю файл: name={}, mime={}, size={}",
        document.file_name,
        document.mime_type,
        document.file_size,
    )
    file_info = bot.get_file(document.file_id)
    payload = bot.download_file(file_info.file_path)
    directory = Path(tempfile.mkdtemp(prefix="hay-v2-"))
    path = directory / filename
    path.write_bytes(payload)
    logger.info("Файл сохранён локально: {}, {} байт", path.name, len(payload))
    return path


def remove_download(path: Path) -> None:
    """Удаляет временный каталог со скачанным файлом."""
    shutil.rmtree(path.parent, ignore_errors=True)
    logger.info("Временный файл удалён: {}", path.name)
