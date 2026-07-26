// Google sign-in for the extension. We use chrome.identity.launchWebAuthFlow to
// get a Google ID TOKEN (a signed JWT proving who the user is), then hand it to
// the backend, which verifies it and returns OUR session token.

import { clearToken } from "./storage";
import { authGoogle } from "./api";
import { GOOGLE_CLIENT_ID } from "./config";
import { setToken } from "./storage";

/** The redirect Chrome gives back to the extension: https://<ext-id>.chromiumapp.org/ */
export function redirectUri(): string {
  return chrome.identity.getRedirectURL();
}

export async function signIn(): Promise<void> {
  const nonce = crypto.randomUUID();
  const url = new URL("https://accounts.google.com/o/oauth2/v2/auth");
  url.searchParams.set("client_id", GOOGLE_CLIENT_ID)
  url.searchParams.set("response_type", "id_token")
  url.searchParams.set("redirect_uri", redirectUri());
  url.searchParams.set("scope", "openid email profile");
  url.searchParams.set("nonce", nonce);

  const responseUrl = await chrome.identity.launchWebAuthFlow({
    url: url.toString(),
    interactive: true,
  });

  if (!responseUrl) throw new Error("sign-in cancelled");

  const fragment = new URL(responseUrl).hash.slice(1);
  const idToken = new URLSearchParams(fragment).get("id_token");

  if (!idToken) throw new Error("no id_token in response");

  const jwt = await authGoogle(idToken);
  await setToken(jwt);

}

export async function signOut(): Promise<void> {
  await clearToken();
}
