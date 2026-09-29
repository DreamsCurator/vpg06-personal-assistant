"""Настройка подробных логов в терминал.

Haystack, Docling и HTTP-клиенты пишут в стандартный ``logging``.
Мост перенаправляет эти записи в loguru, чтобы весь вывод бота
был в одном формате.
"""

from __future__ import annotations

import logging
import sys

from loguru import logger

_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """Передаёт записи стандартного logging в loguru.

    Без этого моста строки Haystack и клиентов OpenAI пропали бы
    после перехода бота на loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Пишет одну запись logging в loguru с исходным уровнем."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging() -> None:
    """Включает подробный вывод loguru и перехват стандартного logging.

    Повторный вызов ничего не меняет: обработчики не дублируются.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    _CONFIGURED = True
    logger.info("Логирование включено: loguru и перехват стандартного logging")
