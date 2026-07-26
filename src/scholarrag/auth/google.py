"""Verify Google OAuth ID tokens — proving the caller is who Google says.

The extension gets an ID token from Google (via ``chrome.identity``) and sends
it here. We verify it against Google's public keys and confirm it was issued for
OUR OAuth client id — only then do we trust the identity inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholarrag.config import Settings


@dataclass(frozen=True, slots=True)
class GoogleIdentity:
    """The trusted fields pulled from a verified Google ID token."""

    sub: str  # Google's stable user id (our join key)
    email: str
    name: str | None


class GoogleAuthError(Exception):
    """The Google ID token was missing, invalid, expired, or for another app."""


def verify_google_token(id_token_str: str, settings: Settings) -> GoogleIdentity:
    "Verify a Google ID token and return the identity it certifies."
    if settings.google_client_id is None:
        raise GoogleAuthError("GOOGLE_CLIENT_ID is not configured")

    from google.auth.transport import requests as g_requests
    from google.oauth2 import id_token as google_id_token

    try:
        info = google_id_token.verify_oauth2_token(
            id_token_str, g_requests.Request(), settings.google_client_id
        )
    except Exception as exc:  # ValueError etc. on any failure
        raise GoogleAuthError(str(exc)) from exc

    return GoogleIdentity(sub=info["sub"], email=info["email"], name=info.get("name"))
