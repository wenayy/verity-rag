.PHONY: install corpus ingest test lint eval eval-llm serve docker

install:
	pip install -e ".[dev]"

corpus:
	python scripts/build_corpus.py

ingest:
	verity ingest

test:
	pytest -q

lint:
	ruff check src tests scripts

eval:            ## offline retrieval ablation (no API key)
	verity eval --mode retrieval -k 5

eval-llm:        ## + LLM expansion/rerank ablation and LLM-as-judge answer eval (needs VERITY_LLM_API_KEY)
	verity eval --mode all --with-llm -k 5

serve:
	verity serve

docker:
	docker compose up --build app
