"""The auth security boundary — a FastAPI dependency that resolves the caller.

Any route that declares ``user: User = Depends(get_current_user)`` becomes
protected: no valid ``Authorization: Bearer <jwt>`` header → 401, otherwise it
receives the authenticated ``User`` and can scope its work to them.
"""

from __future__ import annotations

# HTTPException/status, the token helpers, and repo are used by the exercise body
# (the stub below just raises), so they read as "unused" until you implement it.
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from scholarrag.api.deps import get_db
from scholarrag.auth.tokens import InvalidTokenError, decode_access_token
from scholarrag.db import repository as repo
from scholarrag.db.models import User


def get_current_user(request: Request, session: Session = Depends(get_db)) -> User:
    "Resolve the authenticated user from the request's Bearer token."

    header = request.headers.get("Authorization", "")

    if not header.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = header.removeprefix("Bearer ")

    try:
        user_id = decode_access_token(token, request.app.state.settings)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc

    user = repo.get_user(session, user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")
    return user
