"""Session JWTs — WE sign these after Google has verified the user's identity.

Flow: Google proves *who* the user is (see ``google.py``); we then mint our own
short-lived JWT carrying just the user id, and the client sends it back on every
request. Signed with ``settings.jwt_secret`` (HMAC), so only we can issue valid
tokens and any tampering is detected on decode.
"""

from __future__ import annotations

import uuid
from datetime import UTC

from scholarrag.config import Settings

_ALGORITHM = "HS256"  # HMAC-SHA256: symmetric, signed with our secret


class InvalidTokenError(Exception):
    """The session token is missing, malformed, expired, or tampered with."""


def create_access_token(user_id: uuid.UUID, settings: Settings) -> str:
    "Sign a session JWT that carries ``user_id`` and an expiry."
    from datetime import datetime, timedelta

    import jwt

    now = datetime.now(UTC)

    payload = {
        "sub": str(user_id),  # subject = who
        "iat": now,  # issued-at
        "exp": now + timedelta(minutes=settings.jwt_expiry_minutes),  # expires
    }

    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> uuid.UUID:
    "Verify a session JWT's signature + expiry and return its user id."
    import jwt

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc

    return uuid.UUID(payload["sub"])
