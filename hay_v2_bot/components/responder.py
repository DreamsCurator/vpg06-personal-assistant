"""Компонент генерации ответа: память, документы и инструменты агента."""

from __future__ import annotations

from haystack import Document, component
from haystack.components.agents import Agent
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from loguru import logger

from components.openai_factory import chat_generator
from components.prompts import SYSTEM_PROMPT
from components.settings import Settings
from components.tools import build_tools


def format_memories(documents: list[Document]) -> str:
    """Собирает найденные реплики памяти в текст для системного промпта."""
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


def format_documents(documents: list[Document]) -> str:
    """Собирает чанки файлов с именем, страницей и номером чанка."""
    lines: list[str] = []
    for document in documents:
        content = (document.content or "").strip()
        if not content:
            continue
        file_name = document.meta.get("file_name") or "документ"
        details: list[str] = []
        page = document.meta.get("page_number")
        chunk_index = document.meta.get("chunk_index")
        if page is not None:
            details.append(f"стр. {int(float(page))}")
        if chunk_index is not None:
            details.append(f"чанк {int(float(chunk_index))}")
        location = f", {', '.join(details)}" if details else ""
        lines.append(f"[{file_name}{location}]\n{content}")
    if not lines:
        return "По этому запросу сохранённых документов нет."
    return "\n\n".join(lines)


def _log_hits(title: str, documents: list[Document]) -> None:
    """Пишет в терминал, сколько фрагментов нашёл ретривер и с каким score."""
    if not documents:
        logger.info("{}: ничего не найдено", title)
        return
    logger.info("{}: {} фрагментов", title, len(documents))
    for document in documents:
        score = None if document.score is None else round(float(document.score), 3)
        logger.info(
            "  source={} file={} page={} chunk={} role={} score={}",
            document.meta.get("source"),
            document.meta.get("file_name"),
            document.meta.get("page_number"),
            document.meta.get("chunk_index"),
            document.meta.get("role"),
            score,
        )


@component
class AssistantResponder:
    """Отвечает пользователю агентом Haystack с учётом памяти и документов.

    Ретриверы пайплайна передают сюда уже найденные фрагменты.
    Системный промпт собирает ChatPromptBuilder, затем агент может вызвать
    факт о кошках или фото собаки.
    """

    def __init__(self, settings: Settings) -> None:
        """Запоминает настройки и инструменты на всё время работы бота."""
        self._settings = settings
        self._tools = build_tools(settings)

    @component.output_types(answer=str)
    def run(
        self,
        query: str,
        documents: list[Document],
        memories: list[Document],
        profile: str,
        history: list[ChatMessage],
    ) -> dict[str, str]:
        """Собирает промпт и возвращает текст ответа агента.

        Параметры:
            query: Текущая реплика пользователя.
            documents: Чанки файлов этого пользователя, ближайшие к реплике.
            memories: Фрагменты долговременной памяти диалога.
            profile: Строка с именем и username из Telegram.
            history: Короткая история текущего чата без текущей реплики.

        Возвращает:
            Словарь с ключом ``answer``. Если модель промолчала, там
            короткая просьба написать ещё раз.
        """
        _log_hits("Retriever документов", documents)
        _log_hits("Retriever памяти", memories)
        prompt_builder = ChatPromptBuilder(
            template=[ChatMessage.from_system(SYSTEM_PROMPT)],
            required_variables="*",
        )
        prompt = prompt_builder.run(
            PROFILE=profile,
            MEMORY=format_memories(memories),
            DOCUMENTS=format_documents(documents),
        )["prompt"]
        system_text = prompt[0].text or ""
        agent = Agent(
            chat_generator=chat_generator(self._settings),
            tools=self._tools,
            system_prompt=system_text,
            exit_conditions=["text"],
            max_agent_steps=8,
            raise_on_tool_invocation_failure=False,
        )
        logger.info("Агент запущен, реплика {} символов, история {} сообщений", len(query), len(history))
        result = agent.run(messages=[*history, ChatMessage.from_user(query)])
        last_message = result.get("last_message")
        answer = ((last_message.text if last_message is not None else "") or "").strip()
        if not answer:
            logger.warning("Агент вернул пустой ответ")
            answer = "Я не смог сформулировать ответ. Напишите ещё раз."
        logger.info("Ответ агента готов, {} символов", len(answer))
        return {"answer": answer}
