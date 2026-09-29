"""Инструменты агента: факт о кошках и фото собаки с описанием породы.

Фото, которое нужно показать в Telegram, складывается в ContextVar.
Haystack вызывает инструменты в пуле потоков и переносит туда contextvars,
поэтому список снимков виден и в обработчике сообщения.
"""

from __future__ import annotations

import base64
import contextvars
import json
import urllib.error
import urllib.request

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.tools import tool
from haystack.utils import Secret
from loguru import logger

from components.prompts import VISION_PROMPT
from components.settings import CAT_FACT_URL, DOG_IMAGE_URL, Settings

pending_photos: contextvars.ContextVar[list[tuple[str, bytes]] | None] = contextvars.ContextVar(
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
    photos = pending_photos.get()
    if photos is None:
        logger.warning("Фото получено вне обработки сообщения и в чат не попадёт: {}", url)
        return
    photos.append((url, data))
    logger.info("Фото поставлено в очередь Telegram: {}, {} байт", url, len(data))


def build_tools(settings: Settings) -> list:
    """Собирает инструменты агента, замкнутые на модель и прямой API OpenAI.

    Описание породы тоже идёт в OpenAI через ``settings.openai_base_url``,
    той же чат-моделью, что и основной диалог.
    """

    @tool
    def random_cat_fact() -> str:
        """Возвращает один случайный факт о кошках с бесплатного API catfact.ninja.

        Вызывай, когда пользователь просит факт, интересность или короткую новость о кошках
        и не просит фотографию.
        """
        logger.info("Инструмент random_cat_fact: запрос {}", CAT_FACT_URL)
        try:
            payload = _get_json(CAT_FACT_URL)
            body = payload.get("fact")
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            logger.warning("Факт о кошках недоступен: {}", error)
            return "Сейчас не получилось получить факт о кошках. Попробуй попросить ещё раз чуть позже."
        if not isinstance(body, str) or not body.strip():
            logger.warning("Сервис фактов о кошках вернул пустой ответ: {}", payload)
            return "Сервис фактов о кошках вернул пустой ответ."
        logger.info("Факт о кошках получен, {} символов", len(body.strip()))
        return body.strip()

    @tool
    def random_dog_photo() -> str:
        """Берёт случайное фото собаки с dog.ceo, загружает его в модель OpenAI и описывает породу.

        Вызывай, когда пользователь просит фото, картинку, породу или «покажи собаку».
        Не придумывай ссылку сам: инструмент получает снимок сам.
        """
        logger.info("Инструмент random_dog_photo: запрос {}", DOG_IMAGE_URL)
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
            vision = OpenAIChatGenerator(
                api_key=Secret.from_token(settings.openai_api_key),
                model=settings.chat_model,
                api_base_url=settings.openai_base_url,
            )
            logger.info(
                "Описываю фото собаки моделью {} через {}",
                settings.chat_model,
                settings.openai_base_url,
            )
            result = vision.run(messages=[ChatMessage.from_user(content_parts=[VISION_PROMPT, image])])
            description = (result["replies"][0].text or "").strip()
        except Exception as error:
            logger.exception("Не удалось описать фото собаки")
            return (
                "Фото скачано и будет показано в чате, но модель не смогла его описать: "
                f"{error}"
            )
        if not description:
            return "Фото будет показано в чате, но модель вернула пустое описание породы."
        logger.info("Описание породы получено, {} символов", len(description))
        return (
            "Фото уже передано в чат. Перескажи пользователю это описание породы и её историю, "
            f"не меняя породу:\n{description}"
        )

    return [random_cat_fact, random_dog_photo]
