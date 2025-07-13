from typing import Optional

import pandas as pd
import numpy as np

from model import TopicVectorizerClusterizer
from model import TitleClassifier


class BookFinder:
    def __init__(self,
                 min_cluster_size: int,
                 cluster_selection_epsilon: float,
                 k_neighbours_inference: int = 5,
                 umap_neighbors: int = 15,
                 umap_min_dist: float = 0.1,
                 umap_metric: str = 'euclidian'
                 ) -> None:
        self._tvc = TopicVectorizerClusterizer(min_cluster_size=min_cluster_size,
                                               cluster_selection_epsilon=cluster_selection_epsilon,
                                               k_neighbours_inference=k_neighbours_inference,
                                               umap_neighbors=umap_neighbors,
                                               umap_min_dist=umap_min_dist,
                                               umap_metric=umap_metric)
        self._title_classifier = TitleClassifier()

        self._threshold = None
        self._counter = 0

    def fit(self, input_path: str, num_of_books: int = None) -> None:
        data = pd.read_csv(input_path)

        self._tvc.fit_transform(data)
        self._title_classifier.load_data(data, num_of_books)
        self._title_classifier.data['cluster'] = self._tvc.vectors['cluster']
        self._title_classifier.load_model()

        count = self._tvc.data.shape[0]
        self._threshold = count * 2
        self._counter += count

    def predict(self, info: str, save_path: Optional[str] = None) -> pd.DataFrame:
        # TODO: updating the samples lists (for Clustering model and TitleClassifier)
        sample = self._tvc.predict(description=info)
        cluster = sample.get('cluster').values[0]
        return self._title_classifier.get_titles(info, save_path, cluster)
