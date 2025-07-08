from model import TopicVectorizerClusterizer

from sklearn.metrics import silhouette_score

import optuna
from airflow import DAG
from airflow.decorators import task, task_group
from airflow.dates import days_ago


def model_performance(data: str, cluser_size: int, cluster_eps: float, cluster_kneigh: int) -> float:
    # TODO: change data argument type from str to DataFrame (the logic is somewhat poor with the current approach)
    tvc = TopicVectorizerClusterizer(data, cluser_size, cluster_eps, cluster_kneigh)
    tvc.process_vector_databases()
    tvc.clusterize_descriptions()  # TODO: shouldn`t it return the dataset??? now we just rely on the internal property
    score = silhouette_score(tvc.vectors_info)  # TODO: incorrect
    return score


def objective(data: str, trial: optuna.trial.Trial) -> float:
    cluster_sizes = trial.suggest_uniform('min_cluster_sizes', 1, 1000)
    cluster_epsilons = trial.suggest_loguniform('cluster_epsilons', 1e-5, 100)
    cluster_kneighbors = trial.suggest_int('cluster_kneighbors', 1, 20)
    return 0.42  # TODO: return the metric value
