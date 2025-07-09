from model import TopicVectorizerClusterizer

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import optuna
from airflow import DAG
from airflow.decorators import task, task_group
from airflow.dates import days_ago


def model_performance(data: pd.DataFrame, cluser_size: int, cluster_eps: float, cluster_kneigh: int) -> float:
    tvc = TopicVectorizerClusterizer(cluser_size, cluster_eps, cluster_kneigh)
    result = tvc.fit_transform(data)
    score = silhouette_score(np.stack(result['vector'].values), result['cluster'].values)
    return score


def objective(data: pd.DataFrame, trial: optuna.trial.Trial) -> float:
    cluster_sizes = trial.suggest_uniform('min_cluster_sizes', 1, 1000)
    cluster_epsilons = trial.suggest_loguniform('cluster_epsilons', 1e-5, 100)
    cluster_kneighbors = trial.suggest_int('cluster_kneighbors', 1, 20)
    return model_performance(data, cluster_sizes, cluster_epsilons, cluster_kneighbors)
