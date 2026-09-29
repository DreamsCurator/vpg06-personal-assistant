# Telegram-бот v2

Вторая версия персонального помощника. Диалог, память и инструменты первой версии сохранены. К ним добавлен разбор файлов через Docling и поиск по их чанкам в Pinecone.

Ядро — пайплайны Haystack. Эмбеддинги считает удалённая модель OpenAI `text-embedding-3-small` через API OpenAI. Локальная модель эмбеддингов не используется. Токенайзер `tiktoken` только нарезает текст на чанки.

## Что умеет бот

Кнопка меню слева от поля ввода:

| Команда | Действие |
| --- | --- |
| `/start`, `/help` | Приветствие и список команд |
| `/cat` | Случайный факт о кошке с [catfact.ninja](https://catfact.ninja/) |
| `/dog` | Случайное фото собаки с [dog.ceo](https://dog.ceo/dog-api/). Снимок уходит в OpenAI, бот называет породу и кратко рассказывает, как она появилась |

Обычный текст запускает `generation_pipeline`:

1. Запрос эмбеддится моделью `text-embedding-3-small`.
2. Ретривер ищет чанки файлов этого пользователя.
3. Второй ретривер ищет долговременную память диалога.
4. Агент отвечает с учётом найденного и при необходимости вызывает инструменты.

Новые реплики по-прежнему пишутся в Pinecone. Порог сходства `0.85`: тот же текст пропускается, близкая другая формулировка обновляет слот, далёкая создаёт новый.

Документ (PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, CSV или изображение) запускает `ingestion_pipeline`:

1. Бот пишет: «Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…»
2. Docling разбирает файл и режет его на чанки.
3. У каждого чанка в метаданных есть имя файла, номер чанка и номер страницы, если Docling его нашёл.
4. Чанки эмбеддятся напрямую в OpenAI и сохраняются в Pinecone.
5. Бот пишет: «Готово. Я изучил этот файл, теперь можем его обсудить.»
6. Следом уходит одно предложение — краткое резюме файла.

Дальнейшие вопросы по файлу идут через тот же `generation_pipeline`. Память и файлы разных людей не смешиваются: поиск фильтруется по `user_id`.

## Структура

```
hay_v2_bot/
  main.py                 # сборка пайплайнов и запуск бота
  components/
    settings.py           # .env и модели
    logging_setup.py      # loguru и перехват logging
    openai_factory.py     # чат и эмбеддинги напрямую в OpenAI
    document_store.py     # Pinecone
    chunking.py           # нарезка токенайзером text-embedding-3-small
    metadata.py           # имя файла, номер чанка и страницы
    memory.py             # запись и фильтры памяти диалога
    prompts.py
    tools.py              # факт о кошке и фото собаки
    responder.py          # агент внутри generation pipeline
  pipelines/
    memory_pipelines.py   # запись и поиск реплик
    ingestion.py          # Docling -> эмбеддинг -> Pinecone
    generation.py         # ретриверы документов и памяти -> ответ
    summary.py            # одно предложение о файле
  bot/
    handlers.py           # текст, команды, файлы
    menu.py
    messaging.py
    files.py
    context.py
```

## Запуск

Нужны Python 3.11+, ключи из `.env` в корне проекта и индекс Pinecone размерности 1536. Если индекса нет, бот создаёт serverless-индекс в `us-east-1` с метрикой `cosine`.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Переменные `.env`:

```
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_NAMESPACE=
TELEGRAM_BOT_TOKEN=
```

Чат и эмбеддинги всегда идут на `https://api.openai.com/v1`. Модель эмбеддингов, если её не задать или задать не модель OpenAI, становится `text-embedding-3-small`.

Из корня проекта:

```bash
python -m hay_v2_bot
```

Команда запускает длинный опрос Telegram. Первый запуск импортирует Docling, поэтому старт занимает дольше, чем у первой версии.

Тот же запуск: `python hay_v2_bot/main.py`.

Первый файл скачивает модели разбора Docling. Это распознавание документа, не эмбеддинг. Векторы по-прежнему считает OpenAI.

Прежняя версия бота остаётся в `hay/hay-telegram-bot.py`.
