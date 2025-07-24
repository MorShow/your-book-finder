from utils.clustering_runner import run as clustering_run

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators import PythonOperator

root = os.path.dirname(os.path.abspath(__file__))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


def retrain_cluster_model():
    clustering_run(os.path.join(root, 'data', 'raw', 'gutenberq_books_tiny.csv'), n_trials=20)


with DAG(
        dag_id='night_run',
        default_args=default_args,
        description='Night retrain and tune clustering model',
        schedule_interval='0 2 * * *',  # Runs at 2:00 AM every day
        start_date=datetime.date(datetime.today() - timedelta(days=1)),
        catchup=False,
        tags=['ml', 'optuna', 'clustering']
) as dag:
    retrain_task = PythonOperator(
        task_id='cluster_model_task',
        python_callable=retrain_cluster_model
    )

    retrain_task
