import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from sklearn.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).resolve().parent.parent
location = project_root / 'logs' / 'title_classifier.log'

logging.basicConfig(filename=location, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicVectorizerClusterizer:
    def __init__(self,
                 min_cluster_size: int,
                 cluster_selection_epsilon: float,
                 k_neighbours_inference: int = 5) -> None:
        self._vector_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._cluster_model = HDBSCAN(min_cluster_size=min_cluster_size,
                                      cluster_selection_epsilon=cluster_selection_epsilon)
        self._k_neighbours_inference = k_neighbours_inference
        self._data = None
        self._vectors = None
        self._result = None
        self._titles_processed = []

    @property
    def vector_model(self) -> SentenceTransformer:
        return self._vector_model

    @property
    def data(self) -> pd.DataFrame:
        return self._data

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

    def vectorize(self, input_information: pd.DataFrame | str, text_column: Optional[str] = None) -> np.ndarray:
        if isinstance(text_column, str):
            return self._vector_model.encode(input_information[text_column])
        return self._vector_model.encode(input_information)

    def clusterize_descriptions(self) -> None:
        vectors = self.vectorize(self.data, 'info')
        result = pd.DataFrame({
            'vector': list(vectors),
            'cluster': self._cluster_model.fit_predict(vectors.tolist()).tolist()
        }, index=self._data.index)
        self._vectors = result

    def fit_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self._data = dataset
        self.clusterize_descriptions()
        return self._vectors

    def predict(self, decsription: str) -> pd.DataFrame:
        vector = self.vectorize(decsription)
        vectors_space = list(enumerate(self._vectors['vector'].values))
        vectors_space.sort(key=lambda x: np.linalg.norm(x[1] - vector['vector']))
        vectors_space = vectors_space[:self._k_neighbours_inference]
        clusters = dict()

        for index, _ in vectors_space:
            clusters[self._vectors.iloc[index, 1]] = (
                    clusters.get(self._vectors.iloc[index, 1], 0) + 1)

        result = pd.DataFrame({
            "vector": vector,
            "cluster": max(clusters, key=clusters.get)
        }, index=vector)

        return result


if __name__ == '__main__':
    tvc = TopicVectorizerClusterizer(5, 5, k_neighbours_inference=4)
    train_data = pd.read_csv(
        r'C:\Users\aleks\OneDrive\Desktop\Studying\your-book-finder\data\raw\gutenberq_books_tiny.csv')
    res = tvc.fit_transform(train_data)
    print(res['vector'])
    print(res['cluster'])
