"""Пайплайн одного предложения-резюме по уже разобранному файлу."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from loguru import logger

from components.openai_factory import chat_generator
from components.prompts import SUMMARY_PROMPT
from components.settings import Settings


def build_summary_pipeline(settings: Settings) -> Pipeline:
    """Собирает короткий запрос к чат-модели: ровно одно предложение о файле.

    Эмбеддинги здесь не считаются. Текст уже извлечён Docling,
    модель только пересказывает его напрямую через OpenAI.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=[ChatMessage.from_user(SUMMARY_PROMPT)],
            required_variables=["file_name", "text"],
        ),
    )
    pipeline.add_component("llm", chat_generator(settings))
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    logger.info("Собран summary pipeline: ChatPromptBuilder -> OpenAIChatGenerator")
    return pipeline


def one_sentence(text: str) -> str:
    """Оставляет из ответа модели одно предложение.

    Если модель добавила второе предложение, оно отбрасывается.
    Пустая строка остаётся пустой: вызывающий код подставляет запасную фразу.
    """
    cleaned = " ".join(text.replace("\n", " ").split()).strip().strip('"').strip("«»")
    if not cleaned:
        return ""
    end = len(cleaned)
    for index, char in enumerate(cleaned):
        if char in ".!?" and index > 10:
            end = index + 1
            break
    sentence = cleaned[:end].strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence
