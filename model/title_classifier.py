from constants import MODEL_TITLES_SMALL, MODEL_TITLES_TINY, JSON_FILE_NAME

import random
import datetime
import os
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).resolve().parent.parent
final_path = os.path.join(project_root, 'data', 'final')
log_filename = 'log_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_location = project_root / 'logs' / (log_filename + '.log')

logging.basicConfig(filename=log_location, encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(os.path.dirname(os.path.join(project_root, 'data')), exist_ok=True)


class TitleClassifier:
    def __init__(self, batch_size=20):
        self._model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
        self._json_path = os.path.join(project_root, 'data', JSON_FILE_NAME)
        if not os.path.exists(self._json_path):
            with open(self._json_path, 'w') as f:
                json.dump(dict(), f, indent=4)
        self._json_data = json.load(open(self._json_path))
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self._device)
        self._batch_size = batch_size
        self._data = None
        self._tokenizer, self._model, self._bi_encoder = self.load_model()

    @property
    def data(self):
        return self._data

    @property
    def model_name(self):
        return self._model_name

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return tokenizer, model, bi_encoder

    def load_data(self, data: pd.DataFrame, num_of_books: int = None) -> None:
        # r"../data/raw/gutenberq_books_tiny.csv" - common pipeline
        self._data = data

        if num_of_books is not None:
            self._data = self._data.sample(num_of_books)

    def _title_inference(self,
                         text: str,
                         cluster: Optional[int] = None) -> dict:
        if cluster is not None and 'cluster' in self.data.columns:
            cl_min, cl_max = min(self.data['cluster'].values), max(self.data['cluster'].values)

            if cl_min <= cluster <= cl_max:
                cluster_data = self.data[self.data['cluster'] == cluster]
            else:
                cluster_data = self.data
        else:
            cluster_data = self.data

        cluster_size = cluster_data.shape[0]
        text_embedding = self._bi_encoder.encode(text, convert_to_tensor=True)
        title_options = dict()

        logging.info(f"The model has just started choosing the titles.")

        for _, row in cluster_data.iterrows():
            title = row.get('title')
            text_of_book = row.get('text').split()
            num_of_batches = int(len(text_of_book) / (100 * np.log10(cluster_size) * self.batch_size))

            chunks = []
            for index in range(0, len(text_of_book), 10 * self.batch_size):
                chunks.append(' '.join(text_of_book[index:index + self.batch_size]))

            bi_encoded_chunks = self._bi_encoder.encode(chunks, convert_to_tensor=True)
            bi_encoded_scores = []
            for info, chunk in zip(chunks, bi_encoded_chunks):
                score = cosine_similarity(text_embedding.unsqueeze(0), chunk.unsqueeze(0)).item()
                bi_encoded_scores.append((info, score))
            bi_encoded_scores.sort(key=lambda x: x[1], reverse=True)
            best_chunks = bi_encoded_scores[:num_of_batches]

            title_options[title] = [title for title, _ in best_chunks]

        title_scores = dict()

        for title, chunks in title_options.items():
            if not chunks:
                continue

            input_texts = [(text, chunk) for chunk in chunks]

            encoded = self._tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True
            )

            with torch.no_grad():
                outputs = self._model(**encoded)
                scores = outputs.logits.squeeze(-1)

            avg_score = scores.mean().item()
            title_scores[title] = avg_score

        probabilities = F.softmax(torch.tensor(list(title_scores.values())), dim=-1).tolist()
        title_probabilities = dict()
        for title, prob in zip(title_scores.keys(), probabilities):
            title_probabilities[title] = prob

        return title_probabilities

    def get_titles(self, text: str, save_path=None, cluster: Optional[int] = None) -> pd.DataFrame:
        if self._json_data.get(text, None) is not None:
            return pd.read_csv(os.path.join(final_path, Path(self._json_data[text])))

        inferences = self._title_inference(text, cluster)
        inferences = pd.DataFrame(list(inferences.items()), columns=['title', 'score'])
        top_three_inferences = inferences.sort_values(by='score', ascending=False)[:3]
        if save_path and save_path.endswith('.csv'):
            if save_path in self._json_data.values():
                logger.warning("The csv file with this name already exists, the inference will not be cached.")
            else:
                top_three_inferences.to_csv(os.path.join(final_path, Path(save_path)), index=False)
                self._json_data[text] = save_path
                json.dump(self._json_data, open(os.path.join(project_root, 'data', JSON_FILE_NAME), 'w'), indent=4)

        return top_three_inferences


if __name__ == '__main__':
    from book_finder import BookFinder

    bf = BookFinder(cluster_selection_epsilon=0.2546331876872773,
                    k_neighbours_inference=29,
                    min_cluster_size=17,
                    umap_metric='euclidean',
                    umap_min_dist=0.21872587813809782,
                    umap_neighbors=8,
                    batch_size=55)
    data = pd.read_csv('...')
    bf.fit(data)
    description = ("The story about a mischievous, imaginative, "
                   "and adventurous boy living in a small town on the Mississippi river."
                   "He's known for his cleverness and ability to get himself and his friends "
                   "into and out of trouble through his escapades. While often portrayed as a troublemaker, "
                   "he possesses a good heart and a strong moral compass, ultimately growing into a more "
                   "responsible and empathetic young man")
    print(f"Final answer: {bf.predict(description)}")
