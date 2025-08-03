from model import TitleClassifier

import re
from functools import partial
from typing import Tuple
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize

import mlflow
import mlflow.sklearn
import optuna
from transformers import pipeline
from openai import OpenAI

nltk.download('punkt')


def clean_description(text: str) -> str:
    sentences = sent_tokenize(text)
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        new_words = []

        for i, word in enumerate(words):
            if word.isdigit() or word[0].isupper():
                continue
            new_words.append(word)

        cleaned_sentences.append(" ".join(new_words))

    cleaned_text = " ".join(cleaned_sentences)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


openai_api_key = "empty"
openai_api_base = "http://localhost:8000/v1"


def model_performance(data: pd.DataFrame,
                      batch_size=20) -> None:
    prompt_generator = OpenAI(api_key=openai_api_key,
                              base_url=openai_api_base)

    tc = TitleClassifier(batch_size=batch_size)
    tc.load_model()
    tc.load_data(data)
    tc.data.sample(int(0.2 * tc.data.shape[0]), random_state=42)

    messages = [{
        'role': 'system',
        'content': (
            "You are helping a user describe a book they remember vaguely. "
            "They do not recall the title, author, or any character names. "
            "Your job is to generate a short description based only on the **idea** of the book, "
            "as if the user is describing it to a book recommendation system.\n\n"
            "**Do not** mention the title, author, names of characters, or any specific locations.\n"
            "The description must sound like: 'It's about a boy who...'\n\n"
            "Example:\n"
            "BAD: 'I'm looking for Harry Potter by J.K. Rowling.'\n"
            "GOOD: 'It's about a young boy who discovers he has magical powers and attends a special school.'"
            "Please, write the answer without redundant introductory words, just the description."
        )
    },
        {
            'role': 'user',
            'content': (
                "The book I vaguely remember is titled: Tom Sawyer. "
                "Please help me write a description for a recommendation system, "
                "without including any names, titles, or specific details."
                "Please, write the answer without redundant introductory words, just the description."
            )
        }]

    print(messages)
    answer = prompt_generator.chat.completions.create(
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        messages=messages,
        temperature=0.9
    )
    # print(tc.data.title[:1])
    # print(answer.choices[0].message.content)
    print(clean_description(answer.choices[0].message.content))
    # print(answer)
    # generated_prompts = answer

    # result = result[result['cluster'] != -1]
    # clear_ratio = result.shape[0] / overall_count  # The ratio of clear embeddings
    # noise_ratio = 1 - clear_ratio  # The ratio of noisy embeddings
    # clusters = result['cluster'].values
    #
    # if len(set(clusters)) < 2:
    #     return tvc, {
    #         'silhouette': -1,
    #         'noise_ratio': 1,
    #         'F_score': 0
    #     }
    #
    # vectors = np.stack(result['vector'].values)
    #
    # silhouette = silhouette_score(vectors, clusters, metric='cosine')  # Silhouette score
    # silhouette_scaled = (1 + silhouette) / 2
    #
    # # F-score - our evaluation metric
    # f_score = ((1 + beta**2) * silhouette_scaled * clear_ratio) / (beta**2 * silhouette_scaled + clear_ratio + 10**-8)
    #
    # stats = {
    #     'silhouette': silhouette,
    #     'noise_ratio': noise_ratio,
    #     'F_score': f_score
    # }
    #
    # return tc, stats


# def objective(trial: optuna.trial.Trial, data: pd.DataFrame) -> float:
#     row_count, _ = data.shape
#     cluster_size = int(trial.suggest_discrete_uniform('min_cluster_size', 2, row_count, 3))
#     cluster_epsilon = trial.suggest_loguniform('cluster_epsilon', 1e-5, 100)
#     cluster_kneighbors = trial.suggest_int('cluster_kneighbors', 1, row_count ** 1/2)
#     umap_neigh = int(trial.suggest_discrete_uniform('umap_neigh', 2, 0.25 * row_count, 2))
#     umap_min_dist = trial.suggest_float('umap_min_dist', 0.0, 1.0)
#     umap_metric = trial.suggest_categorical('umap_metric', ['euclidean', 'manhattan',
#                                                                          'chebyshev', 'cosine'])
#
#     model, stats = model_performance(data, cluster_size, cluster_epsilon,
#                                      cluster_kneighbors, umap_neigh, umap_min_dist, umap_metric)
#     score = stats['F_score']
#
#     if score > objective.best_score:
#         objective.best_score = score
#         objective.best_state = {
#             'min_cluster_size': cluster_size,
#             'cluster_epsilon': cluster_epsilon,
#             'cluster_kneighbors': cluster_kneighbors,
#             'silhouette': stats['silhouette'],
#             'noise_ratio': stats['noise_ratio'],
#             'best_score': score,
#             'best_model': model,
#             'best_data': data,
#             'umap_neigh': umap_neigh,
#             'umap_min_dist': umap_min_dist,
#             'umap_metric': umap_metric,
#         }
#
#     return score
#
#
# objective.best_score = -float('inf')
# objective.best_state = {}
#
#
# def run(filename: str, n_trials: int):
#     dataset = pd.read_csv(filename)
#     objective_wrapped = partial(objective,
#                                 data=dataset)
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective_wrapped, n_trials=n_trials)
#
#     tag = filename[filename.find('\\') + 1:filename.rfind('.csv')]
#     with mlflow.start_run(run_name="best_model_run"):
#         mlflow.set_tags({
#             'dataset_name': tag,
#             'timestamp': datetime.now().isoformat(),
#             'phase': 'training'
#         })
#         mlflow.log_param('min_cluster_size', objective.best_state['min_cluster_size'])
#         mlflow.log_param('cluster_epsilon', objective.best_state['cluster_epsilon'])
#         mlflow.log_param('cluster_kneighbors', objective.best_state['cluster_kneighbors'])
#         mlflow.log_param('umap_neighbors', objective.best_state['umap_neigh'])
#         mlflow.log_param('umap_min_dist', objective.best_state['umap_min_dist'])
#         mlflow.log_param('umap_metric', objective.best_state['umap_metric'])
#         mlflow.log_metric('silhouette_score', objective.best_state['silhouette'])
#         mlflow.log_metric('noise_ratio', objective.best_state['noise_ratio'])
#         mlflow.log_metric('F_score', objective.best_score)
#         mlflow.sklearn.log_model(objective.best_state['best_model'],
# #                                  name='best_model',
#                                  input_example=objective.best_state['best_data'])


if __name__ == '__main__':
    table = pd.read_csv(r'...')
    model_performance(table, batch_size=20)
