import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

from torch import Tensor
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).resolve().parent.parent
location = project_root / 'logs' / 'title_classifier.log'

logging.basicConfig(filename=location, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicVectorizer:
    def __init__(self, data_path: str) -> None:
        self._vector_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._data_path = data_path
        self._data = pd.read_csv(self.data_path)
        self._vectors_info = None
        self._vectors_text = None
        self._result = None
        self._titles_processed = None

    @property
    def vector_model(self) -> SentenceTransformer:
        return self._vector_model

    @property
    def data_path(self) -> str:
        return self._data_path

    @property
    def vectors_info(self) -> Tensor:
        return self._vectors_info

    @property
    def vectors_text(self) -> Tensor:
        return self._vectors_text

    @property
    def titles_processed(self) -> Tensor:
        return self._titles_processed

    def display_result_by_title(self, title: str) -> Optional[Tuple[Tuple, int]]:
        titles = self._result['Title'].index

        if self._result is None:
            logging.warning("The dataset is empty, the model have not been running")
            return None
        if title not in titles:
            logging.warning("The title is not in the dataset")
            return None

        vector, cluster = self._result.loc[title, :].tolist()
        return vector, cluster

    def vectorize(self, text: str) -> Tensor:
        vector = self.vector_model.encode(text)
        print(f'Shape of vector: {vector.shape}')
        print(f'Vector: {vector}')
        return vector

    def process_vector_databases(self) -> None:
        for title in self._data['Title'].tolist():
            if title not in self.titles_processed:
                vector_info = self.vectorize(self._data[self._data['Title'] == title]['info'])
                vector_text = self.vectorize(self._data[self._data['Title'] == title]['text'])
                self.vectors_info[title] = vector_info
                self.vectors_text[title] = vector_text


if __name__ == '__main__':
    t = TopicVectorizer(data_path=r'C:\Users\aleks\OneDrive\Desktop\Studying\your-book-finder\data\raw'
                                  r'\gutenberq_books.csv')
    t.vectorize('ewoewoew boook about your your your success!')
