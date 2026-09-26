"""Телеграм-бот-помощник с долговременной памятью в Pinecone.

Бот ведёт диалог через модель OpenAI и хранит сведения о пользователе
в ``PineconeManager``. Перед записью менеджер сравнивает новое сообщение
с уже сохранёнными фрагментами этого пользователя.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict, deque

import telebot
from dotenv import load_dotenv
from openai import OpenAI

from pinecone_manager import PineconeManager

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MEMORY_TOP_K = 6
HISTORY_LIMIT = 8

SYSTEM_PROMPT = """Ты внимательный помощник в Telegram.
Отвечай на языке пользователя, коротко и по делу.
Ниже долговременная память об этом человеке: факты и прошлые реплики.
Опирайся на неё и не выдумывай то, чего в памяти и в текущем диалоге нет.
Если памяти ещё нет, просто общайся. Новые сведения о пользователе сохраняются отдельно."""

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не задан TELEGRAM_BOT_TOKEN.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("assistant-bot")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
chat_client = OpenAI()
memory = PineconeManager()
history: dict[int, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
history_lock = threading.Lock()


def user_metadata(message: telebot.types.Message, role: str) -> dict[str, str]:
    """Собирает поля пользователя, которые кладутся в слот памяти.

    Пустые значения не включаются: Pinecone не использует их в фильтрах.
    ``user_id`` и ``role`` есть всегда, по ним память разных людей не смешивается.
    """
    user = message.from_user
    full_name = " ".join(part for part in (user.first_name, user.last_name) if part)
    metadata = {
        "user_id": str(user.id),
        "chat_id": str(message.chat.id),
        "role": role,
    }
    if user.username:
        metadata["username"] = user.username
    if full_name:
        metadata["full_name"] = full_name
    return metadata


def memory_filter(user_id: int, role: str | None = None) -> dict[str, dict[str, str]]:
    """Фильтр Pinecone: только записи этого пользователя и, если нужно, этой роли."""
    query: dict[str, dict[str, str]] = {"user_id": {"$eq": str(user_id)}}
    if role is not None:
        query["role"] = {"$eq": role}
    return query


def format_memories(matches: list[dict]) -> str:
    """Превращает найденные фрагменты памяти в текст для системного промпта."""
    lines: list[str] = []
    for match in matches:
        meta = match.get("metadata") or {}
        text = str(meta.get("text") or "").strip()
        if not text:
            continue
        speaker = "Пользователь" if meta.get("role") == "user" else "Ассистент"
        lines.append(f"- {speaker}: {text}")
    if not lines:
        return "Пока ничего не сохранено."
    return "\n".join(lines)


def recent_messages(user_id: int) -> list[dict[str, str]]:
    """Возвращает копию короткой истории текущего диалога."""
    with history_lock:
        return list(history[user_id])


def append_history(user_id: int, user_text: str, assistant_text: str) -> None:
    """Добавляет реплики текущего диалога в оперативную историю."""
    with history_lock:
        history[user_id].append({"role": "user", "content": user_text})
        history[user_id].append({"role": "assistant", "content": assistant_text})


def ask_model(user_text: str, memories_text: str, dialogue: list[dict[str, str]]) -> str:
    """Просит чат-модель ответить с учётом памяти и недавних реплик."""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nДолговременная память:\n{memories_text}",
        },
        *dialogue,
        {"role": "user", "content": user_text},
    ]
    response = chat_client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    answer = response.choices[0].message.content
    return (answer or "").strip()


def store_turn(message: telebot.types.Message, user_text: str, assistant_text: str) -> None:
    """Сохраняет реплику пользователя и ответ бота в долговременную память.

    Поиск похожего фрагмента ограничен той же ролью. Ответ бота не затирает
    слова пользователя, даже если формулировки близки.
    """
    user_id = message.from_user.id
    for role, text in (("user", user_text), ("assistant", assistant_text)):
        if not text.strip():
            continue
        result = memory.remember(
            text,
            metadata=user_metadata(message, role),
            metadata_filter=memory_filter(user_id, role),
        )
        logger.info(
            "Память user_id=%s role=%s action=%s similarity=%s",
            user_id,
            role,
            result["action"],
            result["similarity"],
        )


def send_text(message: telebot.types.Message, text: str) -> None:
    """Отправляет ответ, разрезая его на части лимита Telegram."""
    chunks = [text[start : start + 4000] for start in range(0, len(text), 4000)] or [text]
    bot.reply_to(message, chunks[0])
    for chunk in chunks[1:]:
        bot.send_message(message.chat.id, chunk)


@bot.message_handler(commands=["start", "help"])
def start(message: telebot.types.Message) -> None:
    """Приветствует пользователя и объясняет, что бот его запоминает."""
    if message.from_user is None:
        return
    name = message.from_user.first_name or "друг"
    bot.reply_to(
        message,
        (
            f"Привет, {name}. Я помощник и помню, что вы рассказываете о себе.\n"
            "Напишите, чем помочь, или расскажите факт: имя, предпочтения, планы. "
            "В следующем разговоре я смогу к этому вернуться."
        ),
    )


@bot.message_handler(content_types=["text"])
def chat(message: telebot.types.Message) -> None:
    """Отвечает на сообщение и сохраняет сведения о пользователе в Pinecone."""
    if message.from_user is None or not message.text or not message.text.strip():
        return

    user_id = message.from_user.id
    user_text = message.text.strip()
    bot.send_chat_action(message.chat.id, "typing")
    try:
        matches = memory.query_by_text(
            user_text,
            top_k=MEMORY_TOP_K,
            metadata_filter=memory_filter(user_id),
        )
        answer = ask_model(user_text, format_memories(matches), recent_messages(user_id))
        if not answer:
            answer = "Я не смог сформулировать ответ. Напишите ещё раз."
        send_text(message, answer)
        append_history(user_id, user_text, answer)
        store_turn(message, user_text, answer)
    except Exception:
        logger.exception("Не удалось обработать сообщение user_id=%s", user_id)
        bot.reply_to(message, "Не получилось ответить. Попробуйте ещё раз чуть позже.")


@bot.message_handler(func=lambda message: True)
def unsupported(message: telebot.types.Message) -> None:
    """Сообщает, что бот понимает только текст."""
    bot.reply_to(message, "Я понимаю текстовые сообщения. Напишите вопрос словами.")


if __name__ == "__main__":
    logger.info("Бот запущен, модель диалога: %s", CHAT_MODEL)
    bot.infinity_polling(skip_pending=True)
