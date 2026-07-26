"""Auth routes — exchange a Google ID token for our session JWT, and whoami.

``POST /auth/google``: the extension sends the Google ID token it obtained via
``chrome.identity``; we verify it, upsert the user, and return a session JWT.
``GET /auth/me``: a protected route that proves the token works end to end.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from scholarrag.api.deps import get_db
from scholarrag.auth.deps import get_current_user
from scholarrag.auth.google import GoogleAuthError, verify_google_token
from scholarrag.auth.tokens import create_access_token
from scholarrag.db import repository as repo
from scholarrag.db.models import User

router = APIRouter(tags=["auth"])


class GoogleLoginRequest(BaseModel):
    id_token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: str | None


@router.post("/auth/google", response_model=TokenResponse)
async def google_login(
    body: GoogleLoginRequest,
    request: Request,
    session: Session = Depends(get_db),
) -> TokenResponse:
    """Verify the Google ID token, upsert the user, and mint a session JWT."""
    settings = request.app.state.settings
    try:
        identity = verify_google_token(body.id_token, settings)
    except GoogleAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"google auth failed: {exc}"
        ) from exc

    user = repo.upsert_user(
        session, google_sub=identity.sub, email=identity.email, name=identity.name
    )
    return TokenResponse(access_token=create_access_token(user.id, settings))


@router.get("/auth/me", response_model=MeResponse)
async def me(user: User = Depends(get_current_user)) -> MeResponse:
    """Return the authenticated user — the smoke test for the whole auth flow."""
    return MeResponse(id=user.id, email=user.email, name=user.name)
