.PHONY: help build up down shell clean generate-flows generate-fims generate-animation run-workflow download-lake

help:
	@echo "Flood Animation Tool - Make Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make build          Build Docker image"
	@echo "  make setup          Initial setup (copy .env)"
	@echo ""
	@echo "Run:"
	@echo "  make run-workflow   Run complete workflow"
	@echo "  make shell          Open interactive shell in container"
	@echo ""
	@echo "Generate:"
	@echo "  make generate-flows     Generate flow files from NWM data"
	@echo "  make generate-fims      Generate FIM GeoTIFFs"
	@echo "  make generate-animation Generate animation video"
	@echo ""
	@echo "Utilities:"
	@echo "  make download-lake  Download lake polygon (interactive)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove output files"
	@echo "  make clean-all      Remove all generated files and images"

build:
	docker-compose build

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please edit with your credentials"; \
	else \
		echo ".env file already exists"; \
	fi
	@mkdir -p data/input data/output data/cache
	@echo "Directory structure created"

run-workflow:
	docker-compose run --rm flood-animation python run_workflow.py --config config.yaml

shell:
	docker-compose run --rm flood-animation /bin/bash

generate-flows:
	docker-compose run --rm flood-animation python generate_flow_files.py --config config.yaml

generate-fims:
	docker-compose run --rm flood-animation python generate_batch_fims.py --config config.yaml

generate-animation:
	docker-compose run --rm flood-animation python generate_animation.py --config config.yaml

download-lake:
	docker-compose run --rm flood-animation python download_lake_polygon.py --interactive

clean:
	rm -rf data/output/*
	rm -rf data/cache/*
	@echo "Output and cache directories cleaned"

clean-all: clean
	docker-compose down --rmi all -v
	@echo "Docker images and volumes removed"

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f
