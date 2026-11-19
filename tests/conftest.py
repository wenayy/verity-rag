from pathlib import Path

import pytest

from verity.index.embeddings import LSAEmbedder
from verity.index.store import KnowledgeIndex
from verity.ingest.loaders import load_documents

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def docs():
    return load_documents(ROOT / "knowledge_base")


@pytest.fixture(scope="session")
def index(docs):
    return KnowledgeIndex.build(docs, strategy="markdown", embedder=LSAEmbedder(128))
