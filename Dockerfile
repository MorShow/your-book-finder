FROM apache/airflow:3.0.3

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
USER airflow