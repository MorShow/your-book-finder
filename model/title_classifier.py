from constants import MODEL_TITLES_SMALL, MODEL_TITLES_TINY

import os
from pathlib import Path

import pandas as pd
import torch
import nltk
import numpy as np
from transformers import pipeline


class TitleClassifier:
    def __init__(self, titles_list_arg='full', batch_size=20):
        self._model_name = 'facebook/bart-large-mnli'
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._batch_size = batch_size
        self._titles_list_arg = titles_list_arg
        self._titles_list = MODEL_TITLES_SMALL if titles_list_arg == 'small' else MODEL_TITLES_TINY
        self._data = None
        self._model = self.load_model()

    @property
    def titles_list(self):
        return self._titles_list

    @titles_list.setter
    def titles_list(self, titles_list):
        self._titles_list = titles_list

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
        model = pipeline(
            'zero-shot-classification',
            model=self.model_name,
            device=self.device
        )

        return model

    def title_inference(self, text, num_of_batches=None):
        print(text)
        text_sentences = nltk.sent_tokenize(text)

        text_batches = []
        count = 0

        for index in range(0, len(text_sentences), self.batch_size):
            batch = " ".join(text_sentences[index:index + self.batch_size])
            text_batches.append(batch)
            count += 1
            if num_of_batches and count >= num_of_batches:
                break

        output = self.model(
            text_batches,
            self.titles_list
        )

        title_inferences = {}

        for item in output:
            for label, score in zip(item['labels'], item['scores']):
                if label not in title_inferences:
                    title_inferences[label] = []
                title_inferences[label].append(score)

        inferences_means = {key: np.mean(item).item() for key, item in title_inferences.items()}

        return inferences_means

    def get_titles(self, path_to_data, save_path, num_of_books=None, num_of_batches=None):
        if save_path and os.path.exists(save_path):
            self._data = pd.read_csv(save_path, nrows=num_of_books)
            return self.data
        if self._titles_list_arg == 'full':
            self.titles_list = list(self.data.loc[:, 'title'])

        # r"../data/raw/gutenberq_books.csv" - common pipeline
        self._data = pd.read_csv(path_to_data, nrows=num_of_books)
        print(self._data)

        inferences = self.data.loc[:, 'text'].apply(lambda x: self.title_inference(x, num_of_batches))
        inferences = pd.DataFrame.from_dict(inferences).rename(columns={'text': 'scores'})
        save_df = pd.DataFrame(self.data.loc[:, ['title', 'author']])
        save_df[inferences.columns] = inferences[inferences.columns]
        save_df.to_csv(Path(save_path), index=False)

        return save_df
