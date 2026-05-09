.PHONY: help install dev-install test run docker-build docker-run clean lint format

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	uv sync --no-dev

dev-install: ## Install package in development mode with dev dependencies
	uv sync

test: ## Run tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=open_embeddings --cov-report=html --cov-report=term

run: ## Run the development server
	uv run python -m open_embeddings.main

docker-build: ## Build Docker image
	docker build -t open-embeddings .

docker-run: ## Run Docker container
	docker run -p 8765:8765 open-embeddings

docker-compose-up: ## Start services with docker-compose
	docker-compose up --build

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint: ## Run linting
	flake8 open_embeddings/ tests/
	mypy open_embeddings/

format: ## Format code
	black open_embeddings/ tests/
	isort open_embeddings/ tests/

check-format: ## Check code formatting
	black --check open_embeddings/ tests/
	isort --check-only open_embeddings/ tests/

example: ## Run the client example
	uv run python examples/client_example.py

# Development helpers
dev-setup: dev-install ## Setup development environment
	@echo "Development environment set up!"

all-checks: format lint test ## Run all quality checks