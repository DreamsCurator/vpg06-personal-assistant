"""Точка входа второй версии Telegram-бота на Haystack.

Запуск из корня проекта:

    python -m hay_v2_bot
    python hay_v2_bot/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

import telebot
from loguru import logger

from bot.context import BotContext
from bot.handlers import register_handlers
from bot.menu import setup_menu
from components.document_store import build_document_store
from components.logging_setup import configure_logging
from components.settings import load_settings
from pipelines.generation import build_generation_pipeline
from pipelines.ingestion import build_ingestion_pipeline
from pipelines.memory_pipelines import build_memory_index_pipeline, build_memory_query_pipeline
from pipelines.summary import build_summary_pipeline


def main() -> None:
    """Собирает пайплайны, меню команд и запускает длинный опрос Telegram."""
    configure_logging()
    settings = load_settings()
    document_store = build_document_store(settings)
    context = BotContext(
        bot=telebot.TeleBot(settings.telegram_token),
        settings=settings,
        generation_pipeline=build_generation_pipeline(settings, document_store),
        ingestion_pipeline=build_ingestion_pipeline(settings, document_store),
        summary_pipeline=build_summary_pipeline(settings),
        memory_index_pipeline=build_memory_index_pipeline(settings, document_store),
        memory_query_pipeline=build_memory_query_pipeline(settings, document_store),
    )
    register_handlers(context)
    logger.info(
        "Бот v2 собран. Модель диалога: {}. Эмбеддинги: {} через {}. Индекс: {}.",
        settings.chat_model,
        settings.embedding_model,
        settings.openai_base_url,
        settings.pinecone_index,
    )
    setup_menu(context.bot)
    logger.info("Прогреваю клиенты OpenAI в пайплайнах")
    context.memory_index_pipeline.warm_up()
    context.memory_query_pipeline.warm_up()
    context.generation_pipeline.warm_up()
    context.summary_pipeline.warm_up()
    context.ingestion_pipeline.warm_up()
    logger.info("Бот запущен. Ожидаю сообщения и документы.")
    context.bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()
