build-base:
	docker build -f Dockerfile.base -t airflow-base .

build: build-base
	docker-compose build

up:
	docker-compose up -d

rebuild:
	docker-compose build --no-cache && docker-compose up -d --force-recreate