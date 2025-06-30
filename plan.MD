# Project pipeline

### Constants
- PAGES_RATIO:
  - *project developing: 0.0013 (approx. 100 books, 45Mb)*
  - *testing, integration and deployment: 0.3 (approx. 22500 books, 10Gb)*
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
  this vector should be assigned to some cluster later
  4) Inferring: ROBERTA (or another Transformer model) is used for choosing the best point matching the user`s
     query inside the narrowed batch of points

## 4. Model training:
...

## 5. Validation:
...

## 6. API creation:
...