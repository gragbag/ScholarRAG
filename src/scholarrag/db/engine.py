"""Engine and session management.

Design note — *the repository never owns the transaction.* Functions in
``repository.py`` operate on a ``Session`` they're handed and only ``flush``
(never ``commit``). The caller decides the transaction boundary:

* application code uses :func:`session_scope` (commit on success, rollback on error);
* FastAPI routes use :func:`get_db` as a dependency;
* tests wrap each test in a transaction they roll back for isolation.

This is the *Unit of Work* pattern: one clear place per operation where work is
committed, instead of scattered commits deep in helper functions.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from scholarrag.config import Settings, get_settings


def create_db_engine(settings: Settings) -> Engine:
    """Build a SQLAlchemy engine from settings (no globals — handy for tests)."""
    return create_engine(
        settings.postgres_dsn,
        pool_pre_ping=True,  # transparently recover dropped connections
        future=True,
    )


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide default engine (built from env settings once)."""
    return create_db_engine(get_settings())


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, class_=Session)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session with a commit/rollback boundary."""
    with session_scope() as session:
        yield session
