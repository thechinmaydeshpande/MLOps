.PHONY: help install download-data train test lint format docker-build docker-run k8s-deploy clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make download-data   - Download the dataset"
	@echo "  make train          - Train models"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code with black"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-compose - Run with Docker Compose"
	@echo "  make k8s-deploy     - Deploy to Kubernetes"
	@echo "  make clean          - Clean up generated files"

install:
	pip install -r requirements.txt

download-data:
	python src/data_processing/download_data.py

train:
	python src/model/train.py

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src tests --max-line-length=127
	pylint src --exit-zero

format:
	black src tests

docker-build:
	docker build -t heart-disease-api:latest -f deployment/docker/Dockerfile .

docker-run:
	docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

docker-compose:
	cd deployment/docker && docker-compose up -d

docker-stop:
	docker stop heart-disease-api || true
	docker rm heart-disease-api || true

k8s-deploy:
	kubectl apply -f deployment/kubernetes/

k8s-clean:
	kubectl delete -f deployment/kubernetes/ || true

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name 'htmlcov' -delete
	find . -type f -name '.coverage' -delete
	find . -type f -name '*.log' -delete
	rm -rf mlruns/
