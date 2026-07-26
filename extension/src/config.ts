// Where the backend lives, and the Google OAuth client that signs users in.
// For local dev these point at `make run` on :8001. Set your real Google client
// id here after you create it (EXTENSION.md → setup).

export const API_BASE = "http://localhost:8001";

// The OAuth 2.0 Client ID from Google Cloud Console (type: Web application).
// The extension sends Google's ID token to the backend, which verifies it
// against THIS same client id.
export const GOOGLE_CLIENT_ID = "191438034703-0eb2t93o7hs05okp3c3odofro9js34tr.apps.googleusercontent.com";
