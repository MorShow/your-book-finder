import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, HDBSCAN

project_root = Path(__file__).resolve().parent.parent
location = project_root / 'logs' / 'title_classifier.log'

logging.basicConfig(filename=location, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicVectorizerClusterizer:
    def __init__(self,
                 data_path: str,
                 min_cluster_size: int,
                 cluster_selection_epsilon: float,
                 k_neighbours_inference: int = 5) -> None:
        self._vector_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._cluster_model = HDBSCAN(min_cluster_size=min_cluster_size,
                                      cluster_selection_epsilon=cluster_selection_epsilon)
        # self._cluster_model = DBSCAN(min_samples=min_cluster_size,
        #                              eps=cluster_selection_epsilon)
        self._k_neighbours_inference = k_neighbours_inference
        self._data_path = data_path
        self._data = pd.read_csv(self.data_path)
        self._vectors_info = None
        self._vectors_text = None
        self._result = None
        self._titles_processed = []

    @property
    def vector_model(self) -> SentenceTransformer:
        return self._vector_model

    @property
    def data_path(self) -> str:
        return self._data_path

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def vectors_info(self) -> pd.DataFrame:
        return self._vectors_info

    @vectors_info.setter
    def vectors_info(self, vectors_info: pd.DataFrame) -> None:
        self._vectors_info = vectors_info

    @property
    def vectors_text(self) -> pd.DataFrame:
        return self._vectors_text

    @vectors_text.setter
    def vectors_text(self, vectors_text: pd.DataFrame) -> None:
        self._vectors_text = vectors_text

    @property
    def titles_processed(self) -> List[str]:
        return self._titles_processed

    @property
    def k_neighbours_inference(self) -> int:
        return self._k_neighbours_inference

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

    def vectorize(self, table: pd.DataFrame, text_column: str) -> pd.DataFrame:
        vectors = self._vector_model.encode(table[text_column])
        return pd.DataFrame({
            'vector': list(vectors),
        }, index=table['title'])

    def process_vector_databases(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.vectorize(self.data, 'info'), self.vectorize(self.data, 'text')

    def clusterize_descriptions(self) -> None:
        vectors_info, _ = self.process_vector_databases()
        result = pd.DataFrame({
            'vector': vectors_info['vector'],
            'cluster': self._cluster_model.fit_predict(vectors_info['vector'].tolist()).tolist()
        }, index=vectors_info.index)
        self.vectors_info = result

    def predict(self, title: str, decsription: str) -> Tuple[pd.DataFrame, int]:
        vector = self.vectorize(pd.DataFrame({
            'title': [title],
            'info': decsription
        }), 'info')

        vectors_space = list(enumerate(self.vectors_info['vector'].values))
        vectors_space.sort(key=lambda x: np.linalg.norm(x[1] - vector['vector'].values[0]))
        vectors_space = vectors_space[:self._k_neighbours_inference]
        clusters = dict()

        for index, _ in vectors_space:
            clusters[self.vectors_info.iloc[index, 1]] = (
                    clusters.get(self.vectors_info.iloc[index, 1], 0) + 1)

        return vector, max(clusters, key=clusters.get)
