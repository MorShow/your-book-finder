# Project pipeline

### Constants
- PAGES_RATIO:
  - *project developing: 0.0013 (approx. 100 books, 45Mb)*
  - *CI/CD: 0.015 (approx. 1125 books, 500Mb)*
  - *deployment: 0.3 (approx. 22500 books, 10Gb)*
- MODEL_TITLES_SIZE:
  - *project developing: 'full' (all the titles from the dataset)*
  - *testing, integration and deployment: 'small' (some limited predifined list of titles)*
  - *fast checks: 'tiny' (five samples)*
- MODEL_TITLES:
  - *the lists of titles corresponding to the previous set of constants* 
- NUM_OF_BATCHES:
  - *project developing: None (uses all the batches created after sentence tokenization)*
  - *testing and fast checks: 5*

## 1. Data scraping:
- Source: https://www.gutenberg.org/
- Beautiful Soup + requests is not enough => using Scrapy for switching between the pages and dynamic processing will be better
- Parser class
- Some simple unit tests
- Basic unittests: PyTest / Scrapy Contracts (https://docs.scrapy.org/en/latest/topics/contracts.html)

### 1.1 Pipeline:
- Start the parser: scrapy crawl gutenberq -a [pages_ratio=VALUE] -O ../data/raw/gutenberq_books.csv

## 2. Data processing:
...

## 3. Model architecture designing:
- TODO: Pipeline creation
- 1) Embedding: embed the info about the book and the text
  2) Create the clusters based on the embedding vectors (the space of books` descriptions)
  3) Clusterization: train the HDBSCAN model, then the query will be transformed to vector,
  this vector should be assigned to some cluster later.
     Yes, the Qdrant vector Database 
  could be used for this task (but I want my own implementation).
  4) Inferring: cross-encoder is used for choosing the best point matching the user`s
     query inside the narrowed batch of points

### 3.1 Embeddings + Clustering model:
- Embeddings creation - Sentence Transformer (SBERT)
- Clustering via HDBSCAN (classic DBSCAN is poor for the values of the embedding vectors)
- Visualization using t-SNE in 2D space to highlight the quality of the given clusters

## 4. Model training (???):
...

## 5. Parallelization, CI/CD, and Versioning:
- Stack: scrapy-redis, Airflow, and MLFlow
- Clustering: the F-score (an adapted version) as the main metric (maximizing via Optuna search)
  * F-score = ((1 + beta**2) * silhouette_scaled * clear_ratio) / (beta**2 * silhouette_scaled + clear_ratio + 10**-8)
  * The metric scales the Silhoutte score silhouette_scaled = (1 + silhouette_score) / 2
  * clear_ratio - the ratio of correctly classified (not noisy) points
  * The final metric is just a sophisticated combination of them, the user should choose the beta
- Airflow pipeline:
  1. How? -> Working with the Docker container
  2. Commands to work with Docker:
     docker-compose build (build the container, install the dependencies)
     docker-compose up [-d] (start the Airflow server)
     docker-compose down (terminate the container/Airflow server)
     (* or just use make commands)

## 6. Application:
  1. The Gradio—for the process of the model validation.
     For example, in Gradio application you can choose the location of the CSV to cache it
  2. The main application (chatbot) was created in Telegram, so 'pip install python-telegram-bot'
