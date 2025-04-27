import pandas as pd
import torch
import nltk
import numpy as np

from transformers import pipeline


class TitleClassifier:
    def __init__(self, titles_list, batch_size=20):
        self._model_name = 'facebook/bart-large-mnli'
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._batch_size = batch_size
        self._titles_list = titles_list
        self._data = None
        self._model = self.load_model(self.device)

    @property
    def data(self):
        return self._data

    @data.setter
    def load_data(self, path, nrows=None):
        self._data = pd.read_csv(path, nrows=nrows)

    def load_model(self, device):
        model = pipeline(
            'zero-shot-classification',
            model=self.model_name,
            device=device
        )

        return model

    def title_inference(self, text, num_of_batches=None):
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
        # r"../data/raw/gutenberq_books.csv" - common pipeline
        self.load_data(path_to_data, num_of_books)

        inferences = self.data.apply(lambda x: self.title_inference(x, num_of_batches))

        self.data[inferences.columns] = inferences[inferences.columns]
        self.data.to_csv(save_path, index=False)
