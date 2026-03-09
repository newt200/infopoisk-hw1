"""Обратные индексы и поиск по корпусу.

Модуль реализует два блока логики:

1. Реализация с помощью библиотек:
- обратный индекс через частоты;
- обратный индекс через BM25;
- поиск по запросу пользователя.

2. Ручная реализация через словари:
- обратный индекс через частоты;
- обратный индекс через BM25.

Ожидается, что на вход подается DataFrame или CSV-файл с колонкой
`preprocessed_text`, полученной после предобработки.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing_data import RussianTextPreprocessor


@dataclass(slots=True)
class SearchConfig:
    """Конфигурация индексации и поиска.

    Attributes:
        text_column: название колонки с предобработанным текстом.
        lowercase: приводим запрос к нижнему регистру.
        min_token_length: мин. длина токена.
    """

    text_column: str = "preprocessed_text"
    lowercase: bool = True
    min_token_length: int = 2


class QueryPreprocessor:
    """Предобработка поискового запроса."""

    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        self.preprocessor = RussianTextPreprocessor()

    def preprocess(self, query: str) -> list[str]:
        """Применим ту же предобработку, что и к документам."""
        result = self.preprocessor.preprocess_text(query)
        return result["lemmas"]


class CorpusReader:
    """Чтение корпуса из файла."""

    @staticmethod
    def read_csv(input_path: str | Path) -> pd.DataFrame:
        """Прочитаем корпус из CSV-файла."""
        return pd.read_csv(input_path)


class LibraryFrequencyInvertedIndex:
    """Обратный индекс через частоты с помощью CountVectorizer.

    Внутри хранится разреженная матрица document-term, но дополнительно можно
    получить и классический обратный индекс вида:
        term -> {doc_id: tf}
    """

    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.term_document_matrix = None
        self.feature_names: np.ndarray | None = None
        self.inverted_index: dict[str, dict[int, int]] = {}

    def build(self, documents: list[str]) -> None:
        """Посмтроим частотный обратный индекс по корпусу."""
        self.term_document_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.inverted_index = self._build_inverted_index_from_matrix()

    
    def search(self, query_tokens: list[str]) -> dict[int, float]:
        """Найдем документы по частотному индексу.

        Score документа — сумма частот терминов запроса в документе.
        """
        scores = defaultdict(float)

        for token in query_tokens:
            postings = self.inverted_index.get(token, {})

            for doc_id, frequency in postings.items():
                scores[doc_id] += frequency

        return dict(scores)

    def _build_inverted_index_from_matrix(self) -> dict[str, dict[int, int]]:
        """Преобразуем матрицу в словарь обратного индекса."""
        inverted_index: dict[str, dict[int, int]] = {}

        if self.term_document_matrix is None or self.feature_names is None:
            return inverted_index

        matrix_csc = self.term_document_matrix.tocsc() # для доступа к столбцам

        for term_index, term in enumerate(self.feature_names):
            column = matrix_csc[:, term_index]
            doc_ids = column.nonzero()[0] # возвращает ненулевые индексы
            frequencies = column.data # тут уже сами частоты
            inverted_index[term] = {
                int(doc_id): int(frequency)
                for doc_id, frequency in zip(doc_ids, frequencies)
            }

        return inverted_index


class LibraryBM25InvertedIndex:
    """Обратный индекс BM25 с помощью библиотеки rank_bm25."""

    def __init__(self) -> None:
        self.tokenized_documents: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.inverted_index: dict[str, dict[int, int]] = {}

    def build(self, tokenized_documents: list[list[str]]) -> None:
        """Построить BM25-индекс по корпусу."""
        self.tokenized_documents = tokenized_documents
        self.bm25 = BM25Okapi(tokenized_documents)
        self.inverted_index = self._build_inverted_index(tokenized_documents)

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        """Найдем документы по BM25."""
        if self.bm25 is None:
            raise ValueError("Индекс BM25 еще не построен.")

        scores_array = self.bm25.get_scores(query_tokens)
        return {int(doc_id): float(score) for doc_id, score in enumerate(scores_array) if score > 0}

    @staticmethod
    def _build_inverted_index(
        tokenized_documents: list[list[str]],
    ) -> dict[str, dict[int, int]]:
        """Построить словарь обратного индекса для BM25."""
        inverted_index: dict[str, dict[int, int]] = defaultdict(dict)

        for doc_id, tokens in enumerate(tokenized_documents):
            token_counts = Counter(tokens)
            for term, frequency in token_counts.items():
                inverted_index[term][doc_id] = frequency

        return dict(inverted_index)


class ManualFrequencyInvertedIndex:
    """Ручной обратный индекс через частоты.
    В библиотечной использовали CountVectorizer.
    Здесь обычный Counter"""

    def __init__(self) -> None:
        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)

    def build(self, tokenized_documents: list[list[str]]) -> None:
        """Построить индекс через словари."""
        self.inverted_index = defaultdict(dict)

        for doc_id, tokens in enumerate(tokenized_documents):
            token_counts = Counter(tokens)
            for term, frequency in token_counts.items():
                self.inverted_index[term][doc_id] = frequency

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        """Поиск по ручному частотному индексу."""
        scores: defaultdict[int, float] = defaultdict(float)

        for term in query_tokens:
            postings = self.inverted_index.get(term, {})
            for doc_id, frequency in postings.items():
                scores[doc_id] += frequency

        return dict(scores)


class ManualBM25InvertedIndex:
    """Ручной обратный индекс BM25.
    Дополнительно сохраняются длины документов, document frequency и
    средняя длина документа.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1 # вклад частоты терма
        self.b = b # нормализация по длине дока
        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)
        self.document_lengths: dict[int, int] = {}
        self.document_frequencies: dict[str, int] = {}
        self.documents_count: int = 0
        self.average_document_length: float = 0.0

    def build(self, tokenized_documents: list[list[str]]) -> None:
        """Построить ручной BM25-индекс."""
        self.inverted_index = defaultdict(dict)
        self.document_lengths = {}
        self.document_frequencies = {}
        self.documents_count = len(tokenized_documents) # сколько доков в корпусе

        total_length = 0

        for doc_id, tokens in enumerate(tokenized_documents):
            token_counts = Counter(tokens) # считаем частоты терм в этом тексте
            document_length = len(tokens)
            self.document_lengths[doc_id] = document_length
            total_length += document_length 

            for term, frequency in token_counts.items():
                self.inverted_index[term][doc_id] = frequency # строим обр. индекс

        self.average_document_length = (
            total_length / self.documents_count if self.documents_count > 0 else 0.0
        ) # ср. длина дока

        for term, postings in self.inverted_index.items():
            self.document_frequencies[term] = len(postings)

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        """Поиск по ручному BM25-индексу."""
        scores: defaultdict[int, float] = defaultdict(float)

        for term in query_tokens:
            postings = self.inverted_index.get(term, {}) # значения частот для каждого терма
            if not postings:
                continue

            document_frequency = self.document_frequencies[term]
            idf = self._compute_idf(document_frequency)

            for doc_id, term_frequency in postings.items():
                document_length = self.document_lengths[doc_id] # длина текущего дока
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.average_document_length
                )
                scores[doc_id] += idf * numerator / denominator

        return dict(scores)

    def _compute_idf(self, document_frequency: int) -> float:
        """Посчитать IDF для BM25.
        Чем реже слово, тем выше его вес"""
        numerator = self.documents_count - document_frequency + 0.5
        denominator = document_frequency + 0.5
        return math.log(1 + numerator / denominator)


class SearchEngine:
    """Поисковый движок поверх библиотечных обратных индексов."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.query_preprocessor = QueryPreprocessor(self.config)
        self.frequency_index = LibraryFrequencyInvertedIndex()
        self.bm25_index = LibraryBM25InvertedIndex()
        self.dataframe: pd.DataFrame | None = None
        self.documents_as_strings: list[str] = []
        self.documents_as_tokens: list[list[str]] = []

    def fit(self, dataframe: pd.DataFrame) -> None:
        """Построим библиотечные индексы по корпусу."""
        self._validate_dataframe(dataframe)
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.documents_as_strings = self._extract_documents_as_strings(self.dataframe)
        self.documents_as_tokens = [text.split() for text in self.documents_as_strings]

        self.frequency_index.build(self.documents_as_strings)
        self.bm25_index.build(self.documents_as_tokens)

    def search(
        self,
        query: str,
        index_type: str = "bm25",
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Найдем top-k документов по запросу."""
        if self.dataframe is None:
            raise ValueError("Сначала нужно вызвать fit() и построить индексы.")

        query_tokens = self.query_preprocessor.preprocess(query)
        if not query_tokens:
            return self.dataframe.head(0).copy()

        if index_type == "frequency":
            scores = self.frequency_index.search(query_tokens)
        elif index_type == "bm25":
            scores = self.bm25_index.search(query_tokens)
        else:
            raise ValueError("index_type должен быть 'frequency' или 'bm25'.")

        ranked_results = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        if not ranked_results:
            return self.dataframe.head(0).copy()

        doc_ids = [doc_id for doc_id, _ in ranked_results]
        doc_scores = [score for _, score in ranked_results]

        result = self.dataframe.iloc[doc_ids].copy()
        result.insert(0, "doc_id", doc_ids)
        result.insert(1, "score", doc_scores)
        return result

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Проверим, что в корпусе есть нужная колонка."""
        if self.config.text_column not in dataframe.columns:
            raise ValueError(
                f"Колонка '{self.config.text_column}' не найдена во входном DataFrame."
            )

    def _extract_documents_as_strings(self, dataframe: pd.DataFrame) -> list[str]:
        """Получим документы как строки из колонки предобработанного текста."""
        return dataframe[self.config.text_column].fillna("").astype(str).tolist()


class InvertedIndexPipeline:
    """Пайплайн чтения корпуса и построения индексов."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.search_engine = SearchEngine(self.config)

    def fit_from_csv(self, input_path: str | Path) -> SearchEngine:
        """Прочитаем корпус и построить индексы."""
        dataframe = CorpusReader.read_csv(input_path)
        self.search_engine.fit(dataframe)
        return self.search_engine



def build_search_engine(
    input_path: str | Path,
    text_column: str = "preprocessed_text",
) -> SearchEngine:
    """Основная точка входа для пользователя.

    Args:
        input_path: Путь к CSV-файлу с корпусом.
        text_column: Название колонки с предобработанным текстом.

    Returns:
        Готовый поисковый движок.
    """
    config = SearchConfig(text_column=text_column)
    pipeline = InvertedIndexPipeline(config)
    return pipeline.fit_from_csv(input_path)


if __name__ == "__main__":
    engine = build_search_engine(
        input_path="woman_ru_9_topics_preprocessed.csv",
        text_column="preprocessed_text",
    )

    results = engine.search(
        query="Мой муж мне изменил, что делать?",
        index_type="frequency",
        top_k=5,
    )
    print("Результаты поиска:")
    for _, row in results.iterrows():
        print("=" * 80)
        print(f"doc_id: {row['doc_id']}")
        print(f"score : {row['score']:.4f}")
        print()
        print(row["text"])
        print()
