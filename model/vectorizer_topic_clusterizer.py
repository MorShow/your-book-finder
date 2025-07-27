import datetime

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

from umap import UMAP
from sklearn.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).resolve().parent.parent
log_filename = 'log_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_location = project_root / 'logs' / (log_filename + '.log')
vector_model_path = next((Path(__file__).resolve().parent / 'sbert_vectorizer' /
                         'models--sentence-transformers--all-MiniLM-L6-v2' / 'snapshots').iterdir())

logging.basicConfig(filename=log_location, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicVectorizerClusterizer:
    def __init__(self,
                 min_cluster_size: int,
                 cluster_selection_epsilon: float,
                 k_neighbours_inference: int = 5,
                 umap_neighbors: int = 15,
                 umap_min_dist: float = 0.1,
                 umap_metric: str = 'euclidian',
                 umap_components: int = 15) -> None:
        logger.info("TopicVectorizerClusterizer - initializing")
        self._vector_model = SentenceTransformer(str(vector_model_path))
        self._cluster_model = HDBSCAN(min_cluster_size=min_cluster_size,
                                      cluster_selection_epsilon=cluster_selection_epsilon)
        self._dim_reducer = UMAP(n_components=umap_components,
                                 min_dist=umap_min_dist,
                                 n_neighbors=umap_neighbors,
                                 metric=umap_metric,
                                 random_state=42)

        self._k_neighbours_inference = k_neighbours_inference
        self._data = None
        self._vectors = None
        self._descriptions_stack = []

    @property
    def vector_model(self) -> SentenceTransformer:
        return self._vector_model

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def vectors(self) -> pd.DataFrame:
        return self._vectors

    @property
    def k_neighbours_inference(self) -> int:
        return self._k_neighbours_inference

    def display_result_by_description(self, description: str) -> Optional[Tuple[str, np.ndarray, int]]:
        descriptions = list(self.data.index)

        if self.data is None:
            logging.warning("TopicVectorizerClusterizer - The dataset is empty, the model have not been running")
            return None
        if description not in descriptions:
            logging.warning("TopicVectorizerClusterizer - The description is not in the dataset")
            return None

        logging.info("TopicVectorizerClusterizer - the data have been clustered successfully")
        title, vector, cluster = self.data.loc[description, :].tolist()
        return title, vector, cluster

    def vectorize(self, input_information: pd.DataFrame | str, text_column: Optional[str] = None) -> np.ndarray:
        logging.info("TopicVectorizerClusterizer - starting vectorizing the data")
        if isinstance(text_column, str):
            logging.info("TopicVectorizerClusterizer - the column data have been chosen")
            return self._vector_model.encode(input_information[text_column])
        logging.info("TopicVectorizerClusterizer - the book description have been typed by the caller")
        return self._vector_model.encode(input_information)

    def clusterize_descriptions(self) -> None:
        logging.info("TopicVectorizerClusterizer - clustering the descriptions")
        vectors = self._dim_reducer.fit_transform(self.vectorize(self.data, 'info'))
        result = pd.DataFrame({
            'title': self.data['title'].tolist() if 'title' in self.data.columns else self.data.index.tolist(),
            'vector': list(vectors),
            'cluster': self._cluster_model.fit_predict(vectors.tolist()).tolist()
        }, index=self.data['info'])
        self._vectors = result
        logging.info("TopicVectorizerClusterizer - the data have been clustered successfully, "
                     "the dataset have been changed")

    def fit_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        logging.info("TopicVectorizerClusterizer - fitting and transforming the data")
        self._data = dataset
        self.clusterize_descriptions()
        logging.info("TopicVectorizerClusterizer - fit/transform have been done")
        return self._vectors

    def predict(self, description: str) -> pd.DataFrame:
        if description in self._vectors.index.tolist():
            logging.info("TopicVectorizerClusterizer - the input have been clustered previously, "
                         "so the cached result has been used")
            return self._vectors.loc[description, :]

        logging.info("TopicVectorizerClusterizer - starting predicting the input description")
        vector = self.vectorize(description)
        vector = self._dim_reducer.transform(vector.reshape(1, -1))
        vectors_space = list(enumerate(self._vectors['vector'].values))
        vectors_space.sort(key=lambda x: np.linalg.norm(x[1] - vector))
        vectors_space = vectors_space[:self._k_neighbours_inference]
        clusters = dict()

        for index, _ in vectors_space:
            clusters[self._vectors.iloc[index, 2]] = (clusters.get(self._vectors.iloc[index, 2], 0) + 1)

        result = pd.DataFrame({
            'title': [np.nan],
            'vector': [vector.flatten()],
            'cluster': [max(clusters, key=clusters.get)]
        }, index=[description])

        logging.info("TopicVectorizerClusterizer - the description has been clustered successfully")
        return result
