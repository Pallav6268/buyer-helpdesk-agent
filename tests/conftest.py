import pytest

from agent.documents import load_all
from agent.retrieval import HybridRetriever


@pytest.fixture(scope="session")
def retriever():
    # Keyword-only keeps the suite fast and offline; the hybrid path is exercised in the app.
    return HybridRetriever(load_all(), use_embeddings=False)
