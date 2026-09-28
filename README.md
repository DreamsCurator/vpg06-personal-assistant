# Персональный ассистент в Telegram

Телеграм-помощник на Haystack ведёт диалог и хранит его в Pinecone. Перед ответом бот достаёт ближайшие по косинусному сходству фрагменты этого пользователя и продолжает разговор с учётом памяти.

Рядом лежит более ранний вариант без агента: `bot.py` и `pinecone_manager.py`.

## Что умеет Haystack-бот

Кнопка меню в чате:

| Команда | Действие |
| --- | --- |
| `/start`, `/help` | Приветствие и краткая подсказка |
| `/cat` | Случайный факт о кошке с [catfact.ninja](https://catfact.ninja/#/Facts) |
| `/dog` | Случайное фото собаки с [dog.ceo](https://dog.ceo/dog-api/). Снимок уходит в модель OpenAI, бот называет породу и кратко рассказывает, как она появилась |

Обычный текст — это диалог. Новые реплики записываются в Pinecone.

## Как устроена память

Индекс Pinecone подключён через `PineconeDocumentStore` с метрикой `cosine`. Эмбеддинги считает OpenAI `text-embedding-3-small` (1536 измерений). Поиск и запись фильтруются по `user_id`, память разных людей не смешивается.

Граница сходства задаётся в `hay/hay-telegram-bot.py` переменной `COSINE_SIMILARITY_THRESHOLD` (сейчас `0.85`).

- Сходство ниже границы — новая информация, создаётся новый слот.
- Сходство на границе или выше и текст тот же — дубликат, запись пропускается.
- Сходство высокое, но формулировка другая — обновляется существующий слот.

Короткая история текущего чата держится в памяти процесса и передаётся агенту вместе с найденными фрагментами.

## Файлы

| Файл | Назначение |
| --- | --- |
| `hay/hay-telegram-bot.py` | Агент Haystack, меню Telegram, инструменты и память в Pinecone |
| `bot.py` | Прежний телеграм-бот без агента |
| `pinecone_manager.py` | Память прежнего бота |
| `requirements.txt` | Зависимости |

Логи Haystack-бота пишет loguru. Сообщения Haystack и HTTP-клиентов попадают в тот же вывод.

## Запуск

Нужны Python 3.11+, ключи OpenAI, Pinecone и Telegram. Индекс Pinecone должен уже существовать и совпадать по размерности с моделью `text-embedding-3-small` (1536). Если индекса нет, бот создаёт serverless-индекс в `us-east-1` с метрикой `cosine`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Создайте файл `.env` в корне проекта:

```
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_NAMESPACE=
TELEGRAM_BOT_TOKEN=
```

Запуск агента:

```bash
python hay/hay-telegram-bot.py
```

Прежний бот запускается так: `python bot.py`.

Файл `.env` в репозиторий не входит.
