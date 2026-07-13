"""Shared pytest fixtures.

Every test runs against the in-process ``LocalVectorStore`` — no Pinecone key,
no network. We force ``VECTOR_STORE=local`` so the suite is deterministic
regardless of the developer's environment.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from scholarrag.api.main import create_app
from scholarrag.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="ci",
        vector_store="local",
        anthropic_api_key=None,
        pinecone_api_key=None,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
