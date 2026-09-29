"""Отправка сообщений и фото в Telegram."""

from __future__ import annotations

from io import BytesIO

import telebot
from loguru import logger


def send_text(bot: telebot.TeleBot, message: telebot.types.Message, text: str) -> None:
    """Отправляет ответ, разрезая его на части лимита Telegram.

    Первая часть уходит ответом на сообщение пользователя, остальные —
    отдельными сообщениями в тот же чат.
    """
    chunks = [text[start : start + 4000] for start in range(0, len(text), 4000)] or [text]
    bot.reply_to(message, chunks[0])
    for chunk in chunks[1:]:
        bot.send_message(message.chat.id, chunk)


def deliver(
    bot: telebot.TeleBot,
    message: telebot.types.Message,
    text: str,
    photos: list[tuple[str, bytes]],
) -> None:
    """Показывает фото собак, если инструмент их нашёл, и затем текст ответа."""
    if not photos:
        logger.info("В ответе нет фото для Telegram, уходит только текст")
    for url, data in photos:
        photo = BytesIO(data)
        photo.name = "dog.jpg"
        try:
            bot.send_photo(message.chat.id, photo, reply_to_message_id=message.id)
            logger.info("Фото отправлено в Telegram: {}, {} байт", url, len(data))
        except Exception:
            logger.exception("Не удалось отправить фото в Telegram: {}", url)
    send_text(bot, message, text)


def telegram_profile(user: telebot.types.User) -> tuple[str, str | None, str | None]:
    """Имя из профиля Telegram, username и строка для системного промпта."""
    full_name = " ".join(part for part in (user.first_name, user.last_name) if part) or None
    profile = f"Имя: {full_name or 'не указано'}."
    if user.username:
        profile += f" Username: @{user.username}."
    return profile, user.username, full_name
