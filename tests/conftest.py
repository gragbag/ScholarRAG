"""Shared pytest fixtures.

Two families here:
* ``settings`` / ``client`` — force the hermetic config (LocalVectorStore,
  FakeEmbedder) so API tests need no cloud or model.
* ``db_engine`` / ``db`` — a real Postgres, skipped automatically if one isn't
  reachable. Each test runs inside an outer transaction that is always rolled
  back; the session joins it via a SAVEPOINT so an inner rollback (e.g. an
  IntegrityError) doesn't tear down the isolation. Shared here so both the
  repository tests and the pipeline tests can use them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from scholarrag.api.main import create_app
from scholarrag.config import Settings
from scholarrag.db.models import Base


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="ci",
        vector_store="local",
        embedding_provider="fake",  # deterministic, no torch/model download
        anthropic_api_key=None,
        pinecone_api_key=None,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


# A SEPARATE database from the dev one: the fixtures drop/create tables, so they
# must never touch `scholarrag` (which holds seeded/uploaded data). Create it once:
#   docker compose exec postgres psql -U scholarrag -c "CREATE DATABASE scholarrag_test"
TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://scholarrag:scholarrag@localhost:5433/scholarrag_test",
)


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    engine = create_engine(TEST_DSN, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        engine.dispose()
        pytest.skip("Postgres not reachable — start it: docker compose up -d postgres")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db(db_engine: Engine) -> Iterator[Session]:
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
