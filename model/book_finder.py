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
                 umap_metric: str = 'euclidian',
                 umap_components: int = 15,
                 batch_size: int = 20
                 ) -> None:
        self._tvc = TopicVectorizerClusterizer(min_cluster_size=min_cluster_size,
                                               cluster_selection_epsilon=cluster_selection_epsilon,
                                               k_neighbours_inference=k_neighbours_inference,
                                               umap_neighbors=umap_neighbors,
                                               umap_min_dist=umap_min_dist,
                                               umap_metric=umap_metric,
                                               umap_components=umap_components)
        self._title_classifier = TitleClassifier(batch_size=batch_size)

        self._threshold = None
        self._counter = 0

        self._additional_data = dict()

    # def _process_overflow(self) -> None:
    #     additional_df = pd.DataFrame.from_dict(self._additional_data, orient='index')
    #     additional_df.columns = ['title', 'author', 'date', 'info', 'language', 'text', 'cluster']
    #     self.fit(additional_df)

    def fit(self, input: str | pd.DataFrame, num_of_books: int = None) -> None:
        if isinstance(input, str):
            data = pd.read_csv(input)
        else:
            data = input

        self._tvc.fit_transform(data)
        self._title_classifier.load_data(data, num_of_books)
        self._title_classifier.data['cluster'] = self._tvc.vectors['cluster'].values
        self._title_classifier.load_model()

        count = self._tvc.data.shape[0]
        k = 1
        while k * 2 <= count:
            k *= 2

        self._threshold = k * 2
        self._counter += count

    def predict(self, info: str, save_path: Optional[str] = None) -> pd.DataFrame:
        sample = self._tvc.predict(description=info)
        cluster = sample.get('cluster').values[0]
        result = self._title_classifier.get_titles(info, save_path, cluster)

        # # TODO: updating the samples lists (for Clustering model and TitleClassifier)
        # #       and retraining the model
        # # sample: title, author, date, info, language, text, cluster
        # sample = [np.nan, np.nan, np.nan, info, np.nan, np.nan, cluster]
        #
        # self._additional_data[info] = sample
        # self._counter += 1
        #
        # if self._counter >= self._threshold:
        #     self._process_overflow()

        return result
