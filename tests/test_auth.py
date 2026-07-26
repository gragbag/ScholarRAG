"""Auth core tests. Needs the ``auth`` extra (pyjwt) — skips in core-only CI.

``test_upsert_user_is_idempotent`` passes now (repo plumbing); the three skipped
tests are the exercise targets: the JWT pair, get_current_user, Google verify.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

pytest.importorskip("jwt")  # the `auth` extra (PyJWT) — absent in core CI

from scholarrag.config import Settings
from scholarrag.db import repository as repo


def _settings(**over: object) -> Settings:
    return Settings(_env_file=None, **over)  # type: ignore[arg-type]


# ── passes now: the user upsert plumbing ─────────────────────────────────────
def test_upsert_user_is_idempotent(db: Session) -> None:
    u1 = repo.upsert_user(db, google_sub="sub-1", email="a@example.com", name="A")
    db.flush()
    u2 = repo.upsert_user(db, google_sub="sub-1", email="new@example.com")  # same sub
    assert u1.id == u2.id  # same account, not a duplicate
    assert u2.email == "new@example.com"  # email refreshed from the latest sign-in


# ── Exercise 1 — the session JWT pair (auth/tokens.py) ───────────────────────
def test_jwt_roundtrip() -> None:
    import uuid

    from scholarrag.auth.tokens import create_access_token, decode_access_token

    settings = _settings(jwt_secret="unit-test-secret-key-at-least-32-bytes-long")
    uid = uuid.uuid4()
    token = create_access_token(uid, settings)
    assert decode_access_token(token, settings) == uid


def test_jwt_rejects_tampering() -> None:
    import uuid

    from scholarrag.auth.tokens import InvalidTokenError, create_access_token, decode_access_token

    token = create_access_token(
        uuid.uuid4(), _settings(jwt_secret="real-secret-key-at-least-32-bytes-long!!")
    )
    with pytest.raises(InvalidTokenError):  # different secret => bad signature
        decode_access_token(token, _settings(jwt_secret="attacker-secret-key-at-least-32-bytes-xx"))


# ── Exercise 3 — Google token verification (auth/google.py) ──────────────────
def test_verify_google_token(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("google.auth")
    import google.oauth2.id_token as gid

    from scholarrag.auth.google import GoogleAuthError, verify_google_token

    monkeypatch.setattr(
        gid,
        "verify_oauth2_token",
        lambda tok, req, aud: {"sub": "g-123", "email": "u@example.com", "name": "U"},
    )
    identity = verify_google_token("fake-id-token", _settings(google_client_id="my-client"))
    assert identity.sub == "g-123"
    assert identity.email == "u@example.com"

    with pytest.raises(GoogleAuthError):  # no client id configured => can't verify audience
        verify_google_token("fake", _settings(google_client_id=None))


# ── Exercise 2 — the security boundary (auth/deps.py) ────────────────────────
def test_get_current_user_endpoint(db: Session) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from scholarrag.api.deps import get_db
    from scholarrag.api.routes import auth as auth_routes
    from scholarrag.auth.tokens import create_access_token

    settings = _settings(jwt_secret="unit-test-secret-key-at-least-32-bytes-long")
    user = repo.upsert_user(db, google_sub="g-1", email="ann@example.com", name="Ann")
    db.flush()

    app = FastAPI()
    app.state.settings = settings
    app.include_router(auth_routes.router)

    def _db_override() -> object:
        yield db

    app.dependency_overrides[get_db] = _db_override
    client = TestClient(app)

    token = create_access_token(user.id, settings)
    ok = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    assert ok.json()["email"] == "ann@example.com"

    assert client.get("/auth/me").status_code == 401  # no header
    assert client.get("/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
