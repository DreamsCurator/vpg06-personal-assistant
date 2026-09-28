"""Телеграм-помощник на Haystack Agent с памятью в Pinecone.

Диалог ведёт инструмент-вызывающий агент. Каждая реплика пользователя и ответ
помощника пишутся в PineconeDocumentStore. Перед ответом ближайшие фрагменты
этого пользователя читаются по косинусному сходству и попадают в системный
промпт, а короткая история текущего чата передаётся агенту как сообщения.

Инструменты ходят во внешние бесплатные API: случайный факт о кошках и
случайное фото собаки. Фото загружается в модель OpenAI, которая называет породу
и кратко рассказывает, как она появилась.
"""

from __future__ import annotations

import base64
import contextvars
import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
import uuid
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path

import telebot
from dotenv import load_dotenv
from loguru import logger
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import tool
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Граница косинусного сходства для записи в долговременную память.
# Ниже границы — новый фрагмент. На границе и выше тот же текст пропускается,
# другая формулировка перезаписывает найденный слот.
COSINE_SIMILARITY_THRESHOLD = 0.85

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
EMBEDDING_DIMENSION = int(
    os.getenv("OPENAI_EMBEDDING_DIMENSION")
    or EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 1536)
)

MEMORY_TOP_K = 6
HISTORY_LIMIT = 8
CAT_FACT_URL = "https://catfact.ninja/fact"
DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
MEMORY_SOURCE = "haystack-assistant"

SYSTEM_PROMPT = """Ты внимательный персональный помощник в Telegram.
Веди разговор как живой помощник: продолжай уже начатую тему, помни имя, предпочтения и планы этого человека и не переспрашивай то, что уже известно.
Отвечай на языке пользователя, коротко и по делу.
Не выдумывай факты о собеседнике. Бери их только из долговременной памяти и из реплик текущего диалога.
Если памяти ещё нет, просто общайся. Новые сведения сохраняются отдельно.

Инструменты вызывай только когда они нужны:
- random_cat_fact — один случайный факт о кошках.
- random_dog_photo — случайное фото собаки. Инструмент сам скачивает снимок, загружает его в модель OpenAI и возвращает породу и краткую историю породы. Фото в чат отправит бот. Перескажи это описание своими словами и не подменяй породу. Не пиши, что не умеешь показывать картинки.

Профиль в Telegram:
{{PROFILE}}

Долговременная память, ближайшая по косинусному сходству:
{{MEMORY}}
"""

VISION_PROMPT = (
    "На фотографии собака. Определи наиболее вероятную породу или смесь пород. "
    "Кратко расскажи, как такая порода появилась: откуда она, от каких собак её выводили и для какой работы. "
    "Добавь две-три черты внешности, которые видны на снимке. "
    "Если порода неочевидна, так и скажи и назови ближайший вариант. "
    "Пиши по-русски, живо и без заголовков."
)

class _InterceptHandler(logging.Handler):
    """Передаёт записи стандартного logging в loguru.

    Haystack и HTTP-клиенты пишут в logging. Без этого моста их строки
    пропали бы после перехода бота на loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
)
logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

_REQUIRED_ENV = ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "TELEGRAM_BOT_TOKEN")
_missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
if _missing:
    raise ValueError("Не заданы переменные окружения: " + ", ".join(_missing) + ".")

# Индекс Pinecone с метрикой cosine, как в интеграции Haystack.
# dimension нужен только если индекса ещё нет. У text-embedding-3-small это 1536,
# и с такой же размерностью уже собран индекс этого проекта.
document_store = PineconeDocumentStore(
    index=os.environ["PINECONE_INDEX_NAME"],
    namespace=os.getenv("PINECONE_NAMESPACE") or "default",
    metric="cosine",
    dimension=EMBEDDING_DIMENSION,
    spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
    show_progress=False,
)

_embedder_kwargs: dict[str, object] = {"model": EMBEDDING_MODEL}
if EMBEDDING_MODEL.startswith("text-embedding-3"):
    _embedder_kwargs["dimensions"] = EMBEDDING_DIMENSION

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder(**_embedder_kwargs))
indexing_pipeline.add_component(
    "writer",
    DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
)
indexing_pipeline.connect("embedder.documents", "writer.documents")

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", OpenAITextEmbedder(**_embedder_kwargs))
query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

bot = telebot.TeleBot(os.environ["TELEGRAM_BOT_TOKEN"])
history: dict[int, deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
history_lock = threading.Lock()
memory_lock = threading.Lock()
# Haystack вызывает инструменты в пуле потоков и переносит туда contextvars.
# threading.local в этот пул не попадает, поэтому список фото держим в ContextVar.
_pending_photos: contextvars.ContextVar[list[tuple[str, bytes]] | None] = contextvars.ContextVar(
    "pending_dog_photos",
    default=None,
)


def _get_json(url: str) -> dict:
    """Скачивает JSON по публичному URL без ключа."""
    request = urllib.request.Request(url, headers={"User-Agent": "haystack-telegram-assistant"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    if not isinstance(payload, dict):
        raise ValueError(f"Ответ {url} не является JSON-объектом.")
    return payload


def _download(url: str) -> bytes:
    """Скачивает файл по публичному URL."""
    request = urllib.request.Request(url, headers={"User-Agent": "haystack-telegram-assistant"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _queue_photo(url: str, data: bytes) -> None:
    """Кладёт снимок в очередь отправки текущего сообщения."""
    photos = _pending_photos.get()
    if photos is None:
        logger.warning("Фото получено вне обработки сообщения и в чат не попадёт: {}", url)
        return
    photos.append((url, data))
    logger.info("Фото поставлено в очередь Telegram: {}, {} байт", url, len(data))


@tool
def random_cat_fact() -> str:
    """Возвращает один случайный факт о кошках с бесплатного API catfact.ninja.

    Вызывай, когда пользователь просит факт, интересность или короткую новость о кошках
    и не просит фотографию.
    """
    try:
        payload = _get_json(CAT_FACT_URL)
        body = payload.get("fact")
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
        logger.warning("Факт о кошках недоступен: {}", error)
        return "Сейчас не получилось получить факт о кошках. Попробуй попросить ещё раз чуть позже."
    if not isinstance(body, str) or not body.strip():
        return "Сервис фактов о кошках вернул пустой ответ."
    return body.strip()


@tool
def random_dog_photo() -> str:
    """Берёт случайное фото собаки с dog.ceo, загружает его в модель OpenAI и описывает породу.

    Вызывай, когда пользователь просит фото, картинку, породу или «покажи собаку».
    Не придумывай ссылку сам: инструмент получает снимок сам.
    """
    try:
        payload = _get_json(DOG_IMAGE_URL)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        logger.warning("Фото собаки недоступно: {}", error)
        return "Сейчас не получилось скачать случайное фото собаки. Попроси ещё раз чуть позже."

    image_url = payload.get("message")
    if payload.get("status") != "success" or not isinstance(image_url, str) or not image_url.startswith("http"):
        logger.warning("dog.ceo вернул неожиданный ответ: {}", payload)
        return "Сервис случайных фото собак вернул некорректный ответ."

    logger.info("Случайное фото собаки: {}", image_url)
    try:
        image_bytes = _download(image_url)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        logger.warning("Не удалось скачать фото {}: {}", image_url, error)
        return "Ссылка на фото получена, но сам файл скачать не удалось."

    _queue_photo(image_url, image_bytes)
    try:
        image = ImageContent(
            base64_image=base64.b64encode(image_bytes).decode("ascii"),
            detail="high",
        )
        vision = OpenAIChatGenerator(model=CHAT_MODEL)
        result = vision.run(
            messages=[ChatMessage.from_user(content_parts=[VISION_PROMPT, image])]
        )
        description = (result["replies"][0].text or "").strip()
    except Exception as error:
        logger.exception("Не удалось описать фото собаки")
        return (
            "Фото скачано и будет показано в чате, но модель не смогла его описать: "
            f"{error}"
        )
    if not description:
        return "Фото будет показано в чате, но модель вернула пустое описание породы."
    return (
        "Фото уже передано в чат. Перескажи пользователю это описание породы и её историю, "
        f"не меняя породу:\n{description}"
    )


def memory_filter(user_id: int, role: str | None = None) -> dict:
    """Фильтр Haystack: только память этого пользователя, при необходимости одной роли."""
    conditions: list[dict[str, str]] = [
        {"field": "meta.user_id", "operator": "==", "value": str(user_id)},
        {"field": "meta.source", "operator": "==", "value": MEMORY_SOURCE},
    ]
    if role is not None:
        conditions.append({"field": "meta.role", "operator": "==", "value": role})
    return {"operator": "AND", "conditions": conditions}


def retrieve_memories(text: str, user_id: int, top_k: int, role: str | None = None) -> list[Document]:
    """Ищет фрагменты памяти, ближайшие к тексту по косинусному сходству."""
    with memory_lock:
        result = query_pipeline.run(
            {
                "text_embedder": {"text": text},
                "retriever": {
                    "filters": memory_filter(user_id, role),
                    "top_k": top_k,
                },
            }
        )
    return list(result["retriever"]["documents"])


def format_memories(documents: list[Document]) -> str:
    """Собирает найденные фрагменты в текст для системного промпта."""
    lines: list[str] = []
    for document in documents:
        content = (document.content or "").strip()
        if not content:
            continue
        speaker = "Пользователь" if document.meta.get("role") == "user" else "Ассистент"
        lines.append(f"- {speaker}: {content}")
    if not lines:
        return "Пока ничего не сохранено."
    return "\n".join(lines)


def remember(text: str, user_id: int, chat_id: int, role: str, username: str | None, full_name: str | None) -> None:
    """Пишет реплику в Pinecone, если это не повтор уже сохранённого фрагмента.

    Сначала ищется ближайший фрагмент той же роли. Сходство берётся из score
    Pinecone при метрике cosine: чем выше, тем ближе по смыслу.
    """
    if not text.strip():
        return

    matches = retrieve_memories(text, user_id, top_k=1, role=role)
    document_id = str(uuid.uuid4())
    similarity = None
    action = "remembered"
    if matches:
        nearest = matches[0]
        similarity = nearest.score
        same_text = (nearest.content or "").strip() == text.strip()
        if similarity is not None and similarity >= COSINE_SIMILARITY_THRESHOLD:
            if same_text:
                logger.info(
                    "Память user_id={} role={} action=skipped similarity={:.3f}",
                    user_id,
                    role,
                    similarity,
                )
                return
            document_id = nearest.id
            action = "updated"

    meta = {
        "user_id": str(user_id),
        "chat_id": str(chat_id),
        "role": role,
        "source": MEMORY_SOURCE,
    }
    if username:
        meta["username"] = username
    if full_name:
        meta["full_name"] = full_name

    document = Document(id=document_id, content=text.strip(), meta=meta)
    with memory_lock:
        indexing_pipeline.run({"embedder": {"documents": [document]}})
    logger.info(
        "Память user_id={} role={} action={} similarity={}",
        user_id,
        role,
        action,
        None if similarity is None else round(similarity, 3),
    )


def recent_messages(user_id: int) -> list[ChatMessage]:
    """Копия короткой истории текущего диалога."""
    with history_lock:
        return list(history[user_id])


def append_history(user_id: int, user_text: str, assistant_text: str) -> None:
    """Добавляет реплики текущего диалога в оперативную историю."""
    with history_lock:
        history[user_id].append(ChatMessage.from_user(user_text))
        history[user_id].append(ChatMessage.from_assistant(assistant_text))


def build_agent(memory_text: str, profile: str) -> Agent:
    """Собирает агента с памятью этого пользователя в системном промпте.

    Промпт собирается ChatPromptBuilder, как в RAG-пайплайне: шаблон получает
    профиль и найденные фрагменты и отдаёт одно системное сообщение.
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(SYSTEM_PROMPT)],
        required_variables="*",
    )
    prompt = prompt_builder.run(PROFILE=profile, MEMORY=memory_text)["prompt"]
    system_text = prompt[0].text or ""
    return Agent(
        chat_generator=OpenAIChatGenerator(model=CHAT_MODEL),
        tools=[random_cat_fact, random_dog_photo],
        system_prompt=system_text,
        exit_conditions=["text"],
        max_agent_steps=8,
        raise_on_tool_invocation_failure=False,
    )


def telegram_profile(user: telebot.types.User) -> tuple[str, str | None, str | None]:
    """Имя из профиля Telegram, username и строка для системного промпта."""
    full_name = " ".join(part for part in (user.first_name, user.last_name) if part) or None
    profile = f"Имя: {full_name or 'не указано'}."
    if user.username:
        profile += f" Username: @{user.username}."
    return profile, user.username, full_name


def send_text(message: telebot.types.Message, text: str) -> None:
    """Отправляет ответ, разрезая его на части лимита Telegram."""
    chunks = [text[start : start + 4000] for start in range(0, len(text), 4000)] or [text]
    bot.reply_to(message, chunks[0])
    for chunk in chunks[1:]:
        bot.send_message(message.chat.id, chunk)


def deliver(message: telebot.types.Message, text: str, photos: list[tuple[str, bytes]]) -> None:
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
    send_text(message, text)


@bot.message_handler(commands=["start", "help"])
def start(message: telebot.types.Message) -> None:
    """Объясняет, что бот помнит диалог и умеет приносить факты о кошках и фото собак."""
    if message.from_user is None:
        return
    name = message.from_user.first_name or "друг"
    bot.reply_to(
        message,
        (
            f"Привет, {name}. Я помощник и помню, что вы рассказываете: имя, предпочтения, планы.\n"
            "Можно просто поговорить. В меню слева от поля ввода есть команды: "
            "факт о кошке и случайное фото собаки с рассказом о породе."
        ),
    )


@bot.message_handler(commands=["cat"])
def cat_fact(message: telebot.types.Message) -> None:
    """Открывает случайный факт о кошке из кнопки меню."""
    message.text = "Расскажи случайный факт о кошках"
    chat(message)


@bot.message_handler(commands=["dog"])
def dog_photo(message: telebot.types.Message) -> None:
    """Открывает случайное фото собаки и описание породы из кнопки меню."""
    message.text = "Покажи случайное фото собаки и расскажи, что это за порода"
    chat(message)


@bot.message_handler(content_types=["text"])
def chat(message: telebot.types.Message) -> None:
    """Отвечает с учётом памяти и сохраняет новую реплику в Pinecone."""
    if message.from_user is None or not message.text or not message.text.strip():
        return

    user = message.from_user
    user_id = user.id
    user_text = message.text.strip()
    profile, username, full_name = telegram_profile(user)
    photos: list[tuple[str, bytes]] = []
    photos_token = _pending_photos.set(photos)
    bot.send_chat_action(message.chat.id, "typing")
    try:
        try:
            memories = format_memories(retrieve_memories(user_text, user_id, MEMORY_TOP_K))
            agent = build_agent(memories, profile)
            result = agent.run(messages=[*recent_messages(user_id), ChatMessage.from_user(user_text)])
            last_message = result.get("last_message")
            answer = ((last_message.text if last_message is not None else "") or "").strip()
            if not answer:
                answer = "Я не смог сформулировать ответ. Напишите ещё раз."
            logger.info("Инструмент подготовил фото: {}", len(photos))
            deliver(message, answer, photos)
        except Exception:
            logger.exception("Не удалось обработать сообщение user_id={}", user_id)
            bot.reply_to(message, "Не получилось ответить. Попробуйте ещё раз чуть позже.")
            return

        try:
            append_history(user_id, user_text, answer)
            remember(user_text, user_id, message.chat.id, "user", username, full_name)
            remember(answer, user_id, message.chat.id, "assistant", username, full_name)
        except Exception:
            logger.exception("Ответ отправлен, но память не сохранилась user_id={}", user_id)
    finally:
        _pending_photos.reset(photos_token)


@bot.message_handler(func=lambda message: True)
def unsupported(message: telebot.types.Message) -> None:
    """Сообщает, что бот понимает только текст."""
    bot.reply_to(message, "Я понимаю текстовые сообщения. Напишите вопрос словами.")


def setup_menu() -> None:
    """Ставит кнопку меню Telegram на текущие команды бота."""
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


if __name__ == "__main__":
    logger.info(
        "Бот запущен, модель диалога: {}, эмбеддинги: {}, индекс: {}",
        CHAT_MODEL,
        EMBEDDING_MODEL,
        os.environ["PINECONE_INDEX_NAME"],
    )
    setup_menu()
    indexing_pipeline.warm_up()
    query_pipeline.warm_up()
    bot.infinity_polling(skip_pending=True)
