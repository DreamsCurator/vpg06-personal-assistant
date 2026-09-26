"""Чтение и запись в векторный индекс Pinecone.

Документы — это текст: модуль считает эмбеддинг моделью OpenAI
``text-embedding-3-small`` и сохраняет вектор вместе с исходным текстом.
Готовые векторы записываются без повторного эмбеддинга.
Чтение возможно и по готовому вектору, и по тексту запроса.

Долговременная память чат-бота пишется через :meth:`PineconeManager.remember`.
Перед записью считается косинусное сходство с уже сохранёнными фрагментами.
Граница «низкое / высокое» задаётся глобальной переменной
``COSINE_SIMILARITY_THRESHOLD`` в начале этого файла.
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Граница косинусного сходства для долговременной памяти.
# Сходство строго ниже этого числа — новая информация, её запоминаем.
# Сходство на границе и выше — уже известный фрагмент:
# тот же текст пропускаем, другую формулировку записываем в тот же слот.
# Диапазон: от -1 до 1. Выше число — строже критерий «это то же самое».
COSINE_SIMILARITY_THRESHOLD = 0.85

# Модель эмбеддингов OpenAI. Размерность вектора по умолчанию — 1536.
EMBEDDING_MODEL = "text-embedding-3-small"


class MemoryWriteResult(TypedDict):
    """Итог попытки записать сообщение в долговременную память.

    action:
        ``remembered`` — сходство низкое, создан новый слот;
        ``skipped`` — сходство высокое и текст совпал, запись не менялась;
        ``updated`` — сходство высокое, но формулировка другая, слот обновлён.
    id:
        Идентификатор слота, в котором теперь лежит память.
    similarity:
        Косинусное сходство с ближайшим сохранённым фрагментом.
        ``None``, если в пространстве имён ещё ничего нет.
    """

    action: str
    id: str
    similarity: float | None


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Считает косинусное сходство двух векторов.

    Значение 1 — векторы сонаправлены, 0 — ортогональны, -1 — противоположны.
    Эмбеддинги ``text-embedding-3-small`` нормированы, поэтому у близких
    по смыслу текстов сходство обычно лежит между 0 и 1.

    Параметры:
        left: Первый вектор.
        right: Второй вектор той же длины.

    Возвращает:
        Косинус угла между векторами, число от -1 до 1.
        Нулевая длина любого вектора даёт 0.

    Вызывает:
        ValueError: Если векторы разной длины.
    """
    if len(left) != len(right):
        raise ValueError("Нельзя сравнить векторы разной длины.")

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _similarity_threshold() -> float:
    """Читает глобальную границу сходства и проверяет, что она допустима."""
    value = COSINE_SIMILARITY_THRESHOLD
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("COSINE_SIMILARITY_THRESHOLD должен быть числом от -1 до 1.")
    threshold = float(value)
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("COSINE_SIMILARITY_THRESHOLD должен быть в диапазоне от -1 до 1.")
    return threshold


def _chunks(items: Sequence[Any], size: int) -> list[list[Any]]:
    """Режет последовательность на списки длины ``size``.

    Последний кусок может быть короче остальных. Пустая последовательность
    даёт пустой список.
    """
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def _metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Копирует метаданные и выбрасывает ключи со значением ``None``.

    Pinecone не принимает ``null`` в метаданных. ``None`` на входе
    считается пустым словарём.
    """
    return {
        key: value
        for key, value in dict(metadata or {}).items()
        if value is not None
    }


class PineconeManager:
    """Управляет записью документов и векторов и поиском по индексу Pinecone.

    Класс держит клиент Pinecone и клиент OpenAI. Текст документов и текст
    поискового запроса превращаются в векторы моделью ``text-embedding-3-small``.
    Уже посчитанные векторы можно записать и искать без обращения к OpenAI.

    В чат-боте долговременная память пишется методом :meth:`remember`.
    Он сравнивает новое сообщение с ближайшим сохранённым фрагментом и
    решает, создавать слот, пропустить дубликат или обновить вариацию.
    Граница решения — глобальная ``COSINE_SIMILARITY_THRESHOLD``.

    Настройки читаются из окружения, если их не передали в конструктор:

    * ``PINECONE_API_KEY`` — ключ Pinecone;
    * ``PINECONE_INDEX_NAME`` — имя индекса;
    * ``PINECONE_NAMESPACE`` — пространство имён, пустая строка означает
      пространство по умолчанию;
    * ``OPENAI_API_KEY`` — ключ OpenAI;
    * ``OPENAI_EMBEDDING_MODEL`` — модель эмбеддингов, по умолчанию
      ``text-embedding-3-small``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        openai_api_key: str | None = None,
        embedding_model: str | None = None,
        embed_batch_size: int = 64,
        upsert_batch_size: int = 100,
    ) -> None:
        """Открывает индекс Pinecone и клиент OpenAI.

        Параметры:
            api_key: Ключ Pinecone. Если не передан, берётся
                ``PINECONE_API_KEY``.
            index_name: Имя существующего индекса. Если не передано,
                берётся ``PINECONE_INDEX_NAME``.
            namespace: Пространство имён для операций, в которых оно
                не указано отдельно. ``None`` означает значение
                ``PINECONE_NAMESPACE`` или пустую строку.
            openai_api_key: Ключ OpenAI. Если не передан, берётся
                ``OPENAI_API_KEY``.
            embedding_model: Имя модели эмбеддингов. Если не передано,
                берётся ``OPENAI_EMBEDDING_MODEL`` или
                ``text-embedding-3-small``.
            embed_batch_size: Сколько текстов отправлять в одном запросе
                эмбеддингов. Должно быть не меньше 1.
            upsert_batch_size: Сколько векторов отправлять в одном запросе
                записи в Pinecone. Должно быть не меньше 1.

        Вызывает:
            ValueError: Если не задан ключ, имя индекса или размер батча
                меньше 1.
        """
        load_dotenv()

        pinecone_api_key = api_key or os.getenv("PINECONE_API_KEY")
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.namespace = namespace if namespace is not None else (os.getenv("PINECONE_NAMESPACE") or "")
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL") or EMBEDDING_MODEL
        self.embed_batch_size = embed_batch_size
        self.upsert_batch_size = upsert_batch_size

        if not pinecone_api_key:
            raise ValueError("Не задан PINECONE_API_KEY.")
        if not self.index_name:
            raise ValueError("Не задан PINECONE_INDEX_NAME.")
        if not openai_key:
            raise ValueError("Не задан OPENAI_API_KEY.")
        if embed_batch_size < 1 or upsert_batch_size < 1:
            raise ValueError("Размер батча должен быть положительным.")

        self._pinecone = Pinecone(api_key=pinecone_api_key)
        self.index = self._pinecone.index(name=self.index_name)
        self._openai = OpenAI(api_key=openai_key)

    def close(self) -> None:
        """Закрывает соединение с индексом Pinecone."""
        self.index.close()

    def _resolve_namespace(self, namespace: str | None) -> str:
        """Возвращает переданное пространство имён или пространство менеджера.

        ``None`` не означает пустую строку: пустая строка — это явное
        пространство по умолчанию, а ``None`` подставляет ``self.namespace``.
        """
        if namespace is None:
            return self.namespace
        return namespace

    def create_embeddings(self, texts: str | Sequence[str]) -> list[list[float]]:
        """Создаёт эмбеддинги моделью OpenAI ``text-embedding-3-small``.

        Строка или список строк отправляются в Embeddings API. Модель
        возвращает плотный вектор размерности 1536 на каждый текст.
        Близкие по смыслу тексты получают близкие векторы, поэтому
        этими векторами потом ищут похожие записи в Pinecone.

        В запрос уходит модель ``self.embedding_model``. По умолчанию это
        ``text-embedding-3-small``; другое имя можно задать в конструкторе
        или переменной ``OPENAI_EMBEDDING_MODEL``.

        Длинный список режется на пачки размера ``embed_batch_size``.
        Ответ API сортируется по полю ``index``, чтобы порядок векторов
        совпал с порядком входных строк, даже если сервер вернул их
        не по порядку.

        Параметры:
            texts: Одна непустая строка или последовательность непустых
                строк. Пробелы по краям допустимы, строка из одних
                пробелов — нет.

        Возвращает:
            Список векторов в том же порядке, что и вход. Каждый вектор —
            список ``float``. У ``text-embedding-3-small`` в нём 1536 чисел.
            Для пустой последовательности возвращается пустой список.
            Одна строка тоже возвращается списком из одного вектора,
            а не голым вектором.

        Вызывает:
            ValueError: Если текст пустой или элемент последовательности
                не является непустой строкой.
            openai.OpenAIError: Если запрос к OpenAI не удался: сеть,
                авторизация, лимит или ошибка API.
        """
        if isinstance(texts, str):
            items = [texts]
        else:
            items = list(texts)
        if not items:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in items):
            raise ValueError("Каждый текст для эмбеддинга должен быть непустой строкой.")

        vectors: list[list[float]] = []
        for chunk in _chunks(items, self.embed_batch_size):
            response = self._openai.embeddings.create(
                model=self.embedding_model,
                input=chunk,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """Создаёт один эмбеддинг моделью OpenAI ``text-embedding-3-small``.

        Удобная обёртка над :meth:`create_embeddings` для одного текста.

        Параметры:
            text: Непустая строка, которую нужно представить вектором.

        Возвращает:
            Вектор из 1536 чисел ``float``.

        Вызывает:
            ValueError: Если ``text`` пустой или состоит из пробелов.
            openai.OpenAIError: Если запрос к OpenAI не удался.
        """
        if not text or not text.strip():
            raise ValueError("Текст для эмбеддинга пуст.")
        return self.create_embeddings(text)[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Создаёт эмбеддинги нескольких строк моделью ``text-embedding-3-small``.

        Поведение совпадает с :meth:`create_embeddings`. Метод оставлен
        как явное имя для пачки текстов.

        Параметры:
            texts: Непустые строки в том порядке, в котором нужны векторы.

        Возвращает:
            Список векторов той же длины и в том же порядке. Пустой вход
            даёт пустой список.

        Вызывает:
            ValueError: Если хотя бы один элемент не является непустой строкой.
            openai.OpenAIError: Если запрос к OpenAI не удался.
        """
        return self.create_embeddings(texts)

    def upsert_vector(
        self,
        vector_id: str,
        values: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
        sparse_values: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> int:
        """Записывает один готовый вектор в индекс.

        Вектор не эмбеддится повторно: в Pinecone уходят уже посчитанные
        числа. Запись с тем же ``id`` в том же пространстве имён
        перезаписывается.

        Параметры:
            vector_id: Уникальный идентификатор записи. Пустая строка
                недопустима.
            values: Плотный вектор. Его длина должна совпадать
                с размерностью индекса. Для ``text-embedding-3-small``
                это 1536, если индекс создан с размерностью по умолчанию.
            metadata: Дополнительные поля записи. Ключи со значением
                ``None`` отбрасываются. Если словарь не передан,
                метаданные не пишутся.
            sparse_values: Разреженный вектор для гибридного индекса,
                словарь с ключами ``indices`` и ``values``.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.

        Возвращает:
            Число записанных векторов. Для одного успешного вызова это 1.

        Вызывает:
            ValueError: Если не задан id или вектор пуст.
        """
        if not vector_id:
            raise ValueError("Не задан id вектора.")
        if not values:
            raise ValueError("Вектор пуст.")

        item: dict[str, Any] = {"id": str(vector_id), "values": list(values)}
        if metadata:
            item["metadata"] = _metadata(metadata)
        if sparse_values:
            item["sparse_values"] = dict(sparse_values)
        return self.upsert_vectors([item], namespace=namespace)

    def upsert_vectors(
        self,
        vectors: Sequence[Any],
        namespace: str | None = None,
    ) -> int:
        """Записывает пачку готовых векторов.

        Каждый элемент передаётся в Pinecone как есть. Допустимые формы:

        * словарь ``{"id": "...", "values": [...], "metadata": {...}}``;
        * кортеж ``(id, values)``;
        * кортеж ``(id, values, metadata)``.

        Список режется на пачки размера ``upsert_batch_size``. Повторный
        ``id`` в том же пространстве имён заменяет старую запись.

        Параметры:
            vectors: Непустая последовательность векторов в одной из форм
                выше.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.

        Возвращает:
            Сколько векторов Pinecone принял во всех пачках.

        Вызывает:
            ValueError: Если список векторов пуст.
        """
        if not vectors:
            raise ValueError("Список векторов пуст.")

        response = self.index.upsert(
            vectors=list(vectors),
            namespace=self._resolve_namespace(namespace),
            batch_size=self.upsert_batch_size,
            show_progress=False,
        )
        return int(response.upserted_count)

    def upsert_document(
        self,
        text: str,
        document_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        text_key: str = "text",
    ) -> int:
        """Эмбеддит один документ и записывает его в индекс.

        Текст превращается в вектор моделью ``text-embedding-3-small``
        и сохраняется вместе с исходной строкой в метаданных.

        Параметры:
            text: Текст документа. Пустая строка недопустима.
            document_id: Идентификатор записи. Если не передан,
                генерируется UUID.
            metadata: Дополнительные поля. Исходный текст записывается
                в ключ ``text_key`` и перекрывает одноимённое поле.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.
            text_key: Ключ метаданных, в который кладётся исходный текст.

        Возвращает:
            Число записанных векторов, для одного документа это 1.

        Вызывает:
            ValueError: Если текст пустой.
            openai.OpenAIError: Если не удалось посчитать эмбеддинг.
        """
        return self.upsert_documents(
            [{"id": document_id, "text": text, "metadata": dict(metadata or {})}],
            namespace=namespace,
            text_key=text_key,
        )

    def upsert_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        namespace: str | None = None,
        text_key: str = "text",
    ) -> int:
        """Эмбеддит документы и записывает их пачками.

        Каждый документ — словарь:

        * ``text`` — обязательный непустой текст;
        * ``id`` — необязательный идентификатор, иначе генерируется UUID;
        * ``metadata`` — необязательный словарь дополнительных полей.

        Тексты эмбеддятся моделью ``text-embedding-3-small`` пачками
        размера ``embed_batch_size``. Исходный текст кладётся в метаданные
        под ключом ``text_key``, чтобы поиск мог вернуть сам документ.

        Параметры:
            documents: Непустая последовательность документов.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.
            text_key: Ключ метаданных для исходного текста.

        Возвращает:
            Сколько векторов записано во всех пачках.

        Вызывает:
            ValueError: Если список пуст или у документа нет непустого
                поля ``text``.
            openai.OpenAIError: Если не удалось посчитать эмбеддинг.
        """
        if not documents:
            raise ValueError("Список документов пуст.")

        prepared: list[dict[str, Any]] = []
        for document in documents:
            text = document.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("У документа должно быть непустое поле text.")
            document_id = document.get("id") or str(uuid.uuid4())
            prepared.append(
                {
                    "id": str(document_id),
                    "text": text,
                    "metadata": {**_metadata(document.get("metadata")), text_key: text},
                }
            )

        upserted = 0
        target_namespace = self._resolve_namespace(namespace)
        for chunk in _chunks(prepared, self.embed_batch_size):
            embeddings = self.embed_texts([item["text"] for item in chunk])
            payload = [
                {
                    "id": item["id"],
                    "values": embedding,
                    "metadata": item["metadata"],
                }
                for item, embedding in zip(chunk, embeddings, strict=True)
            ]
            response = self.index.upsert(
                vectors=payload,
                namespace=target_namespace,
                batch_size=self.upsert_batch_size,
                show_progress=False,
            )
            upserted += int(response.upserted_count)
        return upserted

    def upsert_texts(
        self,
        texts: Sequence[str],
        ids: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        namespace: str | None = None,
        text_key: str = "text",
    ) -> int:
        """Записывает тексты, переданные параллельными списками.

        Это тот же путь, что у :meth:`upsert_documents`: каждый текст
        эмбеддится моделью ``text-embedding-3-small`` и сохраняется
        в индексе. Списки собираются в документы по индексу элемента.

        Параметры:
            texts: Тексты документов.
            ids: Идентификаторы той же длины, что и ``texts``.
                ``None`` — id генерируются автоматически.
            metadatas: Словари метаданных той же длины, что и ``texts``.
                ``None`` — дополнительные поля не задаются.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.
            text_key: Ключ метаданных для исходного текста.

        Возвращает:
            Сколько векторов записано.

        Вызывает:
            ValueError: Если длины ``ids`` или ``metadatas`` не совпадают
                с числом текстов, либо какой-то текст пуст.
            openai.OpenAIError: Если не удалось посчитать эмбеддинг.
        """
        if ids is not None and len(ids) != len(texts):
            raise ValueError("Число id должно совпадать с числом текстов.")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("Число metadata должно совпадать с числом текстов.")

        documents = []
        for index, text in enumerate(texts):
            documents.append(
                {
                    "id": None if ids is None else ids[index],
                    "text": text,
                    "metadata": {} if metadatas is None else dict(metadatas[index]),
                }
            )
        return self.upsert_documents(documents, namespace=namespace, text_key=text_key)

    def query_by_vector(
        self,
        vector: Sequence[float] | None = None,
        top_k: int = 5,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
        sparse_vector: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет ближайшие записи по готовому вектору.

        Запрос не проходит через OpenAI: сравнивается уже посчитанный
        вектор. Результаты упорядочены от наиболее похожего к наименее
        похожему.

        Параметры:
            vector: Плотный вектор запроса той же размерности, что
                и индекс. Можно не передавать, если задан только
                ``sparse_vector``.
            top_k: Сколько ближайших записей вернуть. Не меньше 1.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.
            metadata_filter: Фильтр метаданных Pinecone, например
                ``{"user_id": {"$eq": "42"}}``.
            include_metadata: Возвращать ли метаданные, в том числе
                исходный текст документа.
            include_values: Добавлять ли в ответ сами значения вектора.
            sparse_vector: Разреженный вектор запроса для гибридного
                или разреженного индекса.

        Возвращает:
            Список словарей с ключами ``id``, ``score`` и ``metadata``.
            При ``include_values=True`` у совпадений с вектором есть
            ещё ключ ``values``. Пустой индекс даёт пустой список.

        Вызывает:
            ValueError: Если ``top_k`` меньше 1 или не передан ни плотный,
                ни разреженный вектор.
        """
        if top_k < 1:
            raise ValueError("top_k должен быть не меньше 1.")
        if not vector and not sparse_vector:
            raise ValueError("Для поиска нужен плотный вектор или sparse_vector.")

        response = self.index.query(
            vector=list(vector) if vector else None,
            sparse_vector=dict(sparse_vector) if sparse_vector else None,
            top_k=top_k,
            namespace=self._resolve_namespace(namespace),
            filter=dict(metadata_filter) if metadata_filter else None,
            include_metadata=include_metadata,
            include_values=include_values,
        )
        return self._format_matches(response.matches, include_values)

    def query_by_text(
        self,
        text: str,
        top_k: int = 5,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """Ищет ближайшие записи по тексту запроса.

        Текст сначала превращается в вектор моделью
        ``text-embedding-3-small`` через :meth:`embed_text`, затем
        поиск идёт так же, как в :meth:`query_by_vector`. Запрос и
        документы должны быть эмбеддены одной и той же моделью, иначе
        близость векторов ничего не означает.

        Параметры:
            text: Непустой текст запроса.
            top_k: Сколько ближайших записей вернуть. Не меньше 1.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.
            metadata_filter: Фильтр метаданных Pinecone.
            include_metadata: Возвращать ли метаданные совпадений.
            include_values: Добавлять ли в ответ значения векторов.

        Возвращает:
            Список словарей с ключами ``id``, ``score`` и ``metadata``,
            упорядоченный по убыванию сходства.

        Вызывает:
            ValueError: Если текст пустой или ``top_k`` меньше 1.
            openai.OpenAIError: Если не удалось посчитать эмбеддинг запроса.
        """
        vector = self.embed_text(text)
        return self.query_by_vector(
            vector,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
            include_values=include_values,
        )

    def remember(
        self,
        text: str,
        memory_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        text_key: str = "text",
    ) -> MemoryWriteResult:
        """Записывает сообщение в долговременную память с проверкой на повтор.

        Сначала текст эмбеддится моделью ``text-embedding-3-small``. Затем
        в индексе ищется ближайший уже сохранённый фрагмент, и между его
        вектором и вектором сообщения считается косинусное сходство.

        Решение принимает глобальная ``COSINE_SIMILARITY_THRESHOLD``:

        * сходства нет или оно строго ниже границы — новая информация,
          создаётся новый слот, действие ``remembered``;
        * сходство на границе или выше и текст совпал — дубликат,
          индекс не меняется, действие ``skipped``;
        * сходство на границе или выше, но текст другой — вариация,
          новый вектор и текст записываются в найденный слот,
          действие ``updated``.

        Индекс должен ранжировать соседей по косинусному расстоянию:
        решение опирается на ближайший фрагмент. Чтобы память разных
        пользователей не смешивалась, передайте ``metadata_filter``.

        Параметры:
            text: Сообщение, которое бот хочет запомнить.
            memory_id: Идентификатор нового слота. Используется только
                когда информация новая. Если не передан, генерируется UUID.
                При обновлении сохраняется id уже существующего слота.
            metadata: Дополнительные поля слота, например ``user_id``.
                Исходный текст пишется в ключ ``text_key``.
            namespace: Пространство имён. ``None`` подставляет
                пространство менеджера.
            metadata_filter: Фильтр, среди каких записей искать похожий
                фрагмент. Обычно тот же признак пользователя, что и
                в ``metadata``.
            text_key: Ключ метаданных, где лежит текст фрагмента.
                По нему же отличаем точный дубликат от вариации.

        Возвращает:
            Словарь ``action``, ``id`` и ``similarity``. ``similarity``
            равен ``None``, только если сравнивать было не с чем.

        Вызывает:
            ValueError: Если текст пустой или граница сходства вне диапазона
                от -1 до 1.
            RuntimeError: Если ближайший фрагмент вернулся без вектора
                и сходство посчитать нельзя.
            openai.OpenAIError: Если не удалось посчитать эмбеддинг.
        """
        threshold = _similarity_threshold()
        vector = self.embed_text(text)
        matches = self.query_by_vector(
            vector,
            top_k=1,
            namespace=namespace,
            metadata_filter=metadata_filter,
            include_metadata=True,
            include_values=True,
        )

        if not matches:
            slot_id = str(memory_id or uuid.uuid4())
            self.upsert_vector(
                slot_id,
                vector,
                metadata={**_metadata(metadata), text_key: text},
                namespace=namespace,
            )
            return {"action": "remembered", "id": slot_id, "similarity": None}

        match = matches[0]
        stored_vector = match.get("values")
        if not stored_vector:
            raise RuntimeError(
                "Ближайший фрагмент вернулся без вектора, косинусное сходство посчитать нельзя."
            )
        similarity = cosine_similarity(vector, stored_vector)
        if similarity < threshold:
            slot_id = str(memory_id or uuid.uuid4())
            self.upsert_vector(
                slot_id,
                vector,
                metadata={**_metadata(metadata), text_key: text},
                namespace=namespace,
            )
            return {"action": "remembered", "id": slot_id, "similarity": similarity}

        stored_text = match["metadata"].get(text_key)
        same_text = isinstance(stored_text, str) and stored_text.strip() == text.strip()
        if same_text:
            return {"action": "skipped", "id": str(match["id"]), "similarity": similarity}

        merged = {**match["metadata"], **_metadata(metadata), text_key: text}
        self.upsert_vector(
            str(match["id"]),
            vector,
            metadata=merged,
            namespace=namespace,
        )
        return {"action": "updated", "id": str(match["id"]), "similarity": similarity}

    @staticmethod
    def _format_matches(matches: Sequence[Any], include_values: bool) -> list[dict[str, Any]]:
        """Превращает совпадения Pinecone в обычные словари.

        Параметры:
            matches: Объекты совпадений из ответа ``query``.
            include_values: Добавлять ли координаты вектора, если они
                есть у совпадения.

        Возвращает:
            Список словарей ``id``, ``score``, ``metadata`` и, при
            запросе, ``values``. Порядок совпадений сохраняется.
        """
        results: list[dict[str, Any]] = []
        for match in matches:
            item: dict[str, Any] = {
                "id": match.id,
                "score": match.score,
                "metadata": dict(match.metadata or {}),
            }
            if include_values and match.values is not None:
                item["values"] = list(match.values)
            results.append(item)
        return results
