"""Кнопка меню Telegram и список команд через слэш."""

from __future__ import annotations

import telebot
from loguru import logger


def setup_menu(bot: telebot.TeleBot) -> None:
    """Ставит кнопку меню Telegram на текущие команды бота.

    Команды те же, что у первой версии: старт, справка, факт о кошке
    и случайное фото собаки. Загрузка файла в меню не выносится:
    документ отправляется обычным вложением.
    """
    bot.set_my_commands(
        [
            telebot.types.BotCommand("start", "Начать диалог"),
            telebot.types.BotCommand("help", "Что умеет бот"),
            telebot.types.BotCommand("cat", "Случайный факт о кошке"),
            telebot.types.BotCommand("dog", "Случайное фото собаки"),
        ]
    )
    bot.set_chat_menu_button(menu_button=telebot.types.MenuButtonCommands())
    logger.info("Кнопка меню Telegram обновлена: /start, /help, /cat, /dog")


def start_text(name: str) -> str:
    """Кратко перечисляет команды через слэш и возможность прислать файл."""
    return (
        f"Привет, {name}. Я помощник и помню, что вы рассказываете: имя, предпочтения, планы.\n"
        "Можно просто поговорить или прислать документ: PDF, DOCX и другие файлы. "
        "Я разберу файл, сохраню его и дальше буду учитывать содержимое в ответах.\n"
        "Команды:\n"
        "/start — начать диалог\n"
        "/help — что умеет бот\n"
        "/cat — случайный факт о кошке\n"
        "/dog — случайное фото собаки и рассказ о породе"
    )
