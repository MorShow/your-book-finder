from model import TopicVectorizerClusterizer
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import optuna
# from airflow import DAG
# from airflow.decorators import task, task_group
# from airflow.dates import days_ago


def model_performance(data: pd.DataFrame, cluser_size: int, cluster_eps: float, cluster_kneigh: int) -> float:
    tvc = TopicVectorizerClusterizer(cluser_size, cluster_eps, cluster_kneigh)
    result = tvc.fit_transform(data)
    vectors = np.stack(result['vector'].values)
    clusters = result['cluster'].values

    if len(set(clusters)) < 2:
        return -1
    return silhouette_score(vectors, clusters)


def objective(trial: optuna.trial.Trial, data: pd.DataFrame) -> float:
    row_count, _ = data.shape
    cluster_size = int(trial.suggest_discrete_uniform('min_cluster_size', 2, row_count, 3))
    cluster_epsilon = trial.suggest_loguniform('cluster_epsilon', 1e-5, 100)
    cluster_kneighbor = trial.suggest_int('cluster_kneighbor', 1, row_count ** 1/2)
    return model_performance(data, cluster_size, cluster_epsilon, cluster_kneighbor)


if __name__ == '__main__':
    dataset = pd.read_csv(r'C:\Users\aleks\OneDrive\Desktop\Studying\your-book-finder\data\raw'
                          r'\gutenberq_books_tiny.csv')
    objective_wrapped = partial(objective, data=dataset)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_wrapped, n_trials=100)
