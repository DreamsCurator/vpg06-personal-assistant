"""Связка пайплайнов, истории чата и замка на их запуск."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

import telebot
from haystack import Pipeline
from haystack.dataclasses import ChatMessage

from components.settings import HISTORY_LIMIT, Settings


@dataclass
class BotContext:
    """Всё, что нужно обработчикам Telegram, без глобальных пайплайнов.

    ``pipeline_lock`` не даёт двум апдейтам одновременно запускать
    один и тот же объект Pipeline: Haystack не обещает потокобезопасность
    повторного ``run`` на общем экземпляре.
    """

    bot: telebot.TeleBot
    settings: Settings
    generation_pipeline: Pipeline
    ingestion_pipeline: Pipeline
    summary_pipeline: Pipeline
    memory_index_pipeline: Pipeline
    memory_query_pipeline: Pipeline
    pipeline_lock: threading.Lock = field(default_factory=threading.Lock)
    history_lock: threading.Lock = field(default_factory=threading.Lock)
    history: dict[int, deque[ChatMessage]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
    )

    def recent_messages(self, user_id: int) -> list[ChatMessage]:
        """Копия короткой истории текущего диалога."""
        with self.history_lock:
            return list(self.history[user_id])

    def append_history(self, user_id: int, user_text: str, assistant_text: str) -> None:
        """Добавляет реплики текущего диалога в оперативную историю."""
        with self.history_lock:
            self.history[user_id].append(ChatMessage.from_user(user_text))
            self.history[user_id].append(ChatMessage.from_assistant(assistant_text))
