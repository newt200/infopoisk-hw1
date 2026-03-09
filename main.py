"""Единая точка входа.

Модуль объединяет все этапы пайплайна:
1. предобработка корпуса;
2. построение индексов через библиотеки и словари;
3. построение матричных индексов;
4. поиск по пользовательскому запросу.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from inverted_index import build_search_engine
from matrix_index import build_matrix_search_engine
from preprocessing_data import preprocess_corpus


@dataclass(slots=True)
class PipelineConfig:
    """Конфигурация общего пайплайна.

    Attributes:
        input_path: Путь к исходному корпусу.
        preprocessed_path: Путь к файлу после предобработки.
        text_column: Название колонки с исходным текстом.
        preprocessed_text_column: Название колонки с предобработанным текстом.
    """

    input_path: str | Path = "woman.ru – 9 topic.csv"
    preprocessed_path: str | Path = "woman_ru_9_topics_preprocessed.csv"
    text_column: str = "text"
    preprocessed_text_column: str = "preprocessed_text"


class SearchPipeline:
    """Общий пайплайн проекта.

    Через один объект можно:
    - предобработать корпус;
    - построить нужный тип индекса;
    - выполнить поиск по запросу.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        query: str,
        index_type: str = "bm25",
        implementation: str = "library",
        top_k: int = 10,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """Запустить весь пайплайн поиска.

        Args:
            query: Текст поискового запроса.
            index_type: Тип индекса: "frequency" или "bm25".
            implementation: Вариант реализации:
                - "library" — реализация на библиотеках из inverted_index.py;
                - "matrix" — матричная реализация из matrix_index.py.
            top_k: Сколько результатов вернуть.
            preprocess: Нужно ли сначала запускать предобработку корпуса.

        Returns:
            DataFrame с top-k найденными документами.
        """
        if preprocess:
            self._preprocess_corpus()

        search_engine = self._build_search_engine(implementation)
        return search_engine.search(
            query=query,
            index_type=index_type,
            top_k=top_k,
        )

    def _preprocess_corpus(self) -> pd.DataFrame:
        """Предобработать исходный корпус и сохранить результат."""
        return preprocess_corpus(
            input_path=self.config.input_path,
            output_path=self.config.preprocessed_path,
            text_column=self.config.text_column,
        )

    def _build_search_engine(self, implementation: str):
        """Построить поисковый движок нужного типа."""
        if implementation == "library":
            return build_search_engine(
                input_path=self.config.preprocessed_path,
                text_column=self.config.preprocessed_text_column,
            )

        if implementation == "matrix":
            return build_matrix_search_engine(
                input_path=self.config.preprocessed_path,
                text_column=self.config.preprocessed_text_column,
            )

        raise ValueError(
            "implementation должен быть 'library' или 'matrix'."
        )



def run_search(
    query: str,
    index_type: str = "bm25",
    implementation: str = "library",
    top_k: int = 10,
    preprocess: bool = True,
    input_path: str | Path = "woman.ru – 9 topic.csv",
    preprocessed_path: str | Path = "woman_ru_9_topics_preprocessed.csv",
    text_column: str = "text",
    preprocessed_text_column: str = "preprocessed_text",
) -> pd.DataFrame:
    """Единая точка входа для пользователя.
    Args:
        query: Текст поискового запроса.
        index_type: Тип индекса: "frequency" или "bm25".
        implementation: Тип реализации:
            - "library" — обратные индексы из inverted_index.py;
            - "matrix" — матричные индексы из matrix_index.py.
        top_k: Сколько документов вернуть.
        preprocess: Нужно ли заново запускать предобработку.
        input_path: Путь к исходному корпусу.
        preprocessed_path: Путь к корпусу после предобработки.
        text_column: Название колонки с исходным текстом.
        preprocessed_text_column: Название колонки с предобработанным текстом.

    Returns:
        DataFrame с найденными документами.
    """
    config = PipelineConfig(
        input_path=input_path,
        preprocessed_path=preprocessed_path,
        text_column=text_column,
        preprocessed_text_column=preprocessed_text_column,
    )
    pipeline = SearchPipeline(config)
    return pipeline.run(
        query=query,
        index_type=index_type,
        implementation=implementation,
        top_k=top_k,
        preprocess=preprocess,
    )


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    results = run_search(
        query="Хочу пойти к гадалке",
        index_type="bm25",
        implementation="library",
        top_k=5,
        preprocess=False,
    )

    print("Результаты поиска:")
    for _, row in results.iterrows():
        print("=" * 80)
        print(f"doc_id: {row['doc_id']}")
        print(f"score : {row['score']:.4f}")
        print()
        print(row["text"])
        print()
