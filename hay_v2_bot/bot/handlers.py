"""Обработчики Telegram: диалог, команды и загрузка файлов."""

from __future__ import annotations

import time

import telebot
from haystack import Document
from loguru import logger

from bot.context import BotContext
from bot.files import (
    FileRejected,
    download_telegram_file,
    rejection_reason,
    remove_download,
    supported_formats_text,
)
from bot.menu import start_text
from bot.messaging import deliver, telegram_profile
from components.memory import document_filter, memory_filter, remember
from components.settings import (
    DOCUMENT_TOP_K,
    FILE_READY_TEXT,
    FILE_RECEIVED_TEXT,
    MEMORY_TOP_K,
    SUMMARY_CHAR_LIMIT,
)
from components.tools import pending_photos
from pipelines.summary import one_sentence

_SUMMARY_FALLBACK = "Кратко пересказать файл не удалось, но его текст уже сохранён и его можно обсуждать."


def _preview(documents: list[Document], limit: int) -> str:
    """Склеивает начало чанков, чтобы резюме не упиралось в длину контекста."""
    parts: list[str] = []
    total = 0
    for document in documents:
        content = (document.content or "").strip()
        if not content:
            continue
        room = limit - total
        if room <= 0:
            break
        piece = content if len(content) <= room else content[:room]
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def _message_chunks(text: str) -> list[str]:
    """Режет длинный текст на части лимита Telegram."""
    return [text[start : start + 4000] for start in range(0, len(text), 4000)] or [text]


def _summarize(ctx: BotContext, file_name: str, documents: list[Document]) -> str:
    """Просит summary pipeline одно предложение и отбрасывает всё лишнее."""
    text = _preview(documents, SUMMARY_CHAR_LIMIT)
    if not text.strip():
        return _SUMMARY_FALLBACK
    logger.info("Запускаю summary pipeline для {}, символов текста={}", file_name, len(text))
    started = time.perf_counter()
    try:
        result = ctx.summary_pipeline.run(
            {"prompt_builder": {"file_name": file_name, "text": text}}
        )
        reply = result["llm"]["replies"][0].text or ""
        sentence = one_sentence(reply)
    except Exception:
        logger.exception("Резюме файла {} не построено", file_name)
        return _SUMMARY_FALLBACK
    logger.info("Резюме готово за {:.1f} с: {}", time.perf_counter() - started, sentence or "пусто")
    return sentence or _SUMMARY_FALLBACK


def register_handlers(ctx: BotContext) -> None:
    """Подписывает бота на команды, текст, документы и прочие типы сообщений."""
    bot = ctx.bot

    @bot.message_handler(commands=["start", "help"])
    def start(message: telebot.types.Message) -> None:
        """Объясняет команды через слэш, память диалога и приём файлов."""
        if message.from_user is None:
            return
        name = message.from_user.first_name or "друг"
        logger.info("Команда {} от user_id={}", message.text, message.from_user.id)
        bot.reply_to(message, start_text(name))

    @bot.message_handler(commands=["cat"])
    def cat_fact(message: telebot.types.Message) -> None:
        """Открывает случайный факт о кошке из кнопки меню."""
        logger.info("Команда /cat от user_id={}", getattr(message.from_user, "id", None))
        message.text = "Расскажи случайный факт о кошках"
        chat(message)

    @bot.message_handler(commands=["dog"])
    def dog_photo(message: telebot.types.Message) -> None:
        """Открывает случайное фото собаки и описание породы из кнопки меню."""
        logger.info("Команда /dog от user_id={}", getattr(message.from_user, "id", None))
        message.text = "Покажи случайное фото собаки и расскажи, что это за порода"
        chat(message)

    @bot.message_handler(content_types=["text"])
    def chat(message: telebot.types.Message) -> None:
        """Отвечает с учётом памяти и документов и сохраняет реплику в Pinecone."""
        if message.from_user is None or not message.text or not message.text.strip():
            return

        user = message.from_user
        user_id = user.id
        user_text = message.text.strip()
        profile, username, full_name = telegram_profile(user)
        logger.info(
            "Текст user_id={} chat_id={} символов={}: {}",
            user_id,
            message.chat.id,
            len(user_text),
            user_text[:200],
        )
        photos: list[tuple[str, bytes]] = []
        photos_token = pending_photos.set(photos)
        bot.send_chat_action(message.chat.id, "typing")
        try:
            try:
                started = time.perf_counter()
                with ctx.pipeline_lock:
                    result = ctx.generation_pipeline.run(
                        {
                            "text_embedder": {"text": user_text},
                            "document_retriever": {
                                "filters": document_filter(user_id),
                                "top_k": DOCUMENT_TOP_K,
                            },
                            "memory_retriever": {
                                "filters": memory_filter(user_id),
                                "top_k": MEMORY_TOP_K,
                            },
                            "assistant": {
                                "query": user_text,
                                "profile": profile,
                                "history": ctx.recent_messages(user_id),
                            },
                        }
                    )
                answer = (result["assistant"]["answer"] or "").strip()
                if not answer:
                    answer = "Я не смог сформулировать ответ. Напишите ещё раз."
                logger.info(
                    "generation pipeline завершён за {:.1f} с, фото={}",
                    time.perf_counter() - started,
                    len(photos),
                )
                deliver(bot, message, answer, photos)
            except Exception:
                logger.exception("Не удалось обработать сообщение user_id={}", user_id)
                bot.reply_to(message, "Не получилось ответить. Попробуйте ещё раз чуть позже.")
                return

            try:
                ctx.append_history(user_id, user_text, answer)
                remember(
                    ctx.memory_index_pipeline,
                    ctx.memory_query_pipeline,
                    ctx.pipeline_lock,
                    user_text,
                    user_id,
                    message.chat.id,
                    "user",
                    username,
                    full_name,
                )
                remember(
                    ctx.memory_index_pipeline,
                    ctx.memory_query_pipeline,
                    ctx.pipeline_lock,
                    answer,
                    user_id,
                    message.chat.id,
                    "assistant",
                    username,
                    full_name,
                )
            except Exception:
                logger.exception("Ответ отправлен, но память не сохранилась user_id={}", user_id)
        finally:
            pending_photos.reset(photos_token)

    @bot.message_handler(content_types=["document"])
    def on_document(message: telebot.types.Message) -> None:
        """Разбирает файл через Docling, пишет чанки в Pinecone и присылает резюме."""
        if message.from_user is None or message.document is None:
            return
        user = message.from_user
        logger.info(
            "Документ user_id={} file={} mime={}",
            user.id,
            message.document.file_name,
            message.document.mime_type,
        )
        reason = rejection_reason(message)
        if reason is not None:
            logger.info("Файл отклонён до анализа: {}", reason)
            bot.reply_to(message, reason)
            return
        bot.reply_to(message, FILE_RECEIVED_TEXT)
        path = None
        try:
            path = download_telegram_file(bot, message)
            _, username, full_name = telegram_profile(user)
            display_name = message.document.file_name or path.name
            bot.send_chat_action(message.chat.id, "typing")
            started = time.perf_counter()
            logger.info("Запускаю ingestion pipeline: {}", display_name)
            with ctx.pipeline_lock:
                result = ctx.ingestion_pipeline.run(
                    {
                        "converter": {"sources": [str(path)]},
                        "enricher": {
                            "file_name": display_name,
                            "user_id": str(user.id),
                            "chat_id": str(message.chat.id),
                            "username": username,
                            "full_name": full_name,
                        },
                    },
                    include_outputs_from={"enricher", "writer"},
                )
            documents = list(result.get("enricher", {}).get("documents") or [])
            written = result.get("writer", {}).get("documents_written")
            logger.info(
                "ingestion pipeline завершён за {:.1f} с: чанков={}, записано={}",
                time.perf_counter() - started,
                len(documents),
                written,
            )
            if not documents:
                bot.send_message(
                    message.chat.id,
                    "Не удалось извлечь текст из этого файла. Пришлите другой документ.",
                )
                return
            bot.send_message(message.chat.id, FILE_READY_TEXT)
            for chunk in _message_chunks(_summarize(ctx, display_name, documents)):
                bot.send_message(message.chat.id, chunk)
        except FileRejected as error:
            logger.info("Файл отклонён: {}", error)
            bot.send_message(message.chat.id, str(error))
        except Exception:
            logger.exception("Не удалось обработать файл user_id={}", user.id)
            bot.send_message(message.chat.id, "Не получилось разобрать файл. Попробуйте ещё раз чуть позже.")
        finally:
            if path is not None:
                remove_download(path)

    @bot.message_handler(
        content_types=[
            "photo",
            "audio",
            "video",
            "voice",
            "sticker",
            "animation",
            "video_note",
            "contact",
            "location",
        ]
    )
    def unsupported(message: telebot.types.Message) -> None:
        """Сообщает, что бот ждёт текст или документ поддерживаемого формата."""
        logger.info("Неподдерживаемый тип {} от user_id={}", message.content_type, getattr(message.from_user, "id", None))
        bot.reply_to(
            message,
            "Я понимаю текстовые сообщения и документы: "
            + supported_formats_text()
            + ". Напишите вопрос или пришлите файл.",
        )
