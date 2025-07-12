import pandas as pd
import numpy as np

from vectorizer_topic_clusterizer import TopicVectorizerClusterizer
from title_classifier import TitleClassifier


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

    # TODO: finalize the method
    def predict(self, info: str):
        sample = self._tvc.predict(description=info)
        cluster = sample.get('cluster').values[0]
        self._title_classifier._title_inference(info, cluster)
