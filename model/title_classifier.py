from constants import MODEL_TITLES_SMALL, MODEL_TITLES_TINY, JSON_FILE_NAME

import random
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
location = project_root / 'logs' / 'title_classifier.log'

logging.basicConfig(filename=location, encoding='utf-8', level=logging.INFO)
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

        logging.info(f"The model has just started choosing the best titles.")

        for _, row in cluster_data.iterrows():
            batch_count = 0
            title = row.get('title')
            text_of_book = row.get('text').split()
            num_of_batches = int(len(text_of_book) / (10 * np.log10(cluster_size) * self.batch_size))
            print(f'NUMBER OF BATCHES: {num_of_batches}')

            chunks = []
            for index in range(0, len(text_of_book), self.batch_size):
                chunks.append(' '.join(text_of_book[index:index + self.batch_size]))
            random_chunks = random.sample(chunks, num_of_batches)
            bi_encoded_chunks = self._bi_encoder.encode(random_chunks, convert_to_tensor=True)
            bi_encoded_scores = []

            for info, chunk in zip(chunks, bi_encoded_chunks):
                score = cosine_similarity(text_embedding.unsqueeze(0), chunk.unsqueeze(0)).item()
                bi_encoded_scores.append((info, score))

            topk_batches = int(0.01 * len(bi_encoded_chunks)) + 5 if len(bi_encoded_chunks) > 100 else 5
            bi_encoded_scores.sort(key=lambda x: x[1], reverse=True)
            top_random_chunks = bi_encoded_scores[:topk_batches]
            print(top_random_chunks)

            for chunk, _ in top_random_chunks:
                batch_tokenized = self._tokenizer(
                    text,
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                title_options.setdefault(title, []).append(batch_tokenized)
                batch_count += 1
                if num_of_batches and batch_count >= num_of_batches:
                    break

            print('-' * 40 + 'NEW BOOK' + '-' * 40)

        title_scores = dict()

        for title, tokenized_pairs in title_options.items():
            input_texts = [self._tokenizer.decode(pair_tokenized['input_ids'].squeeze(),
                                                  skip_special_tokens=True)
                           for pair_tokenized in tokenized_pairs]

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
    tc = TitleClassifier()
    tc.load_model()
    data = '...'
    tc.load_data(data)
    description = ("The story about a mischievous, imaginative, "
                   "and adventurous boy living in a small town on the river. "
                   "He's known for his cleverness and ability to get himself and his friends "
                   "into and out of trouble through his escapades. While often portrayed as a troublemaker, "
                   "he possesses a good heart and a strong moral compass, ultimately growing into a more "
                   "responsible and empathetic young man")
    print(f"Final answer: {tc.get_titles(description)}")
