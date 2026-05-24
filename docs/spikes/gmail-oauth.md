# Spike — Gmail OAuth desktop flow

**Ticket**: AONE-201
**Status**: ✅ Validated
**Date**: 2026-05-24

## Goal

Validate the technical path from "user clicks Authorize" to "app receives
authenticated Gmail messages" using the OAuth 2.0 desktop flow. Pin down:

- the minimal scope to request
- the on-disk locations for `credentials.json` and `token.json`
- the refresh-token behavior
- the dependency on Google's Testing-mode quirks

## Outcome

Working end-to-end:

```text
$ uv run python -m aone.gmail.demo
Connecting to Gmail (the browser will open on the first run)…
Connected. Fetching the 5 most recent message IDs…

  1. id=19e5bd539b9a5618
     From:    Google <no-reply@accounts.google.com>
     Subject: Alerta de seguridad
     Date:    Sun, 24 May 2026 21:12:34 GMT
```

`getProfile()` confirmed the authenticated mailbox: `axelvil1999@gmail.com`.

## Architecture

```
src/aone/gmail/auth.py
    └── get_service(credentials_path, token_path) -> Resource
        ├── If token.json exists → load cached creds
        ├── If creds expired but refresh_token present → silent refresh
        └── Else → InstalledAppFlow.run_local_server(port=0)
                  - starts ephemeral local HTTP server on a random port
                  - opens default browser to Google's consent page
                  - blocks until redirect arrives with the auth code
                  - persists creds to token.json
```

## Decisions

### Scope: `gmail.readonly`

Single scope: `https://www.googleapis.com/auth/gmail.readonly`. The app
needs to **read** messages but never send, modify, label, or delete.
Picking the narrowest scope:

- shrinks the blast radius if the token ever leaks
- avoids triggering Google's "restricted scope" verification (`gmail.modify`
  and broader scopes require a security assessment)
- shows the user a much milder consent screen

### File locations

Both `credentials.json` and `token.json` live at the **repo root**, not in
`~/.aone/`. Reasons:

- `credentials.json` is an app-level artifact (same for every user of
  this checkout); keeping it next to the code makes it obvious where to
  drop it during setup.
- `token.json` is a per-user artifact, but in v0 the app is single-user
  and CLI-only — co-locating with the repo simplifies "delete to
  re-auth" flows.
- Both are explicitly gitignored (`.gitignore` lines 22–23).

In v1 (multi-user, server-side), `token.json` moves to a database row
keyed by user ID.

### Local server port: dynamic (`port=0`)

`InstalledAppFlow.run_local_server(port=0)` lets the OS pick any free
port. Avoids the friction of explaining "port 8080 is in use" to users.
Google accepts dynamic ports for `http://localhost` redirect URIs.

### Cache discovery disabled

`build("gmail", "v1", credentials=creds, cache_discovery=False)`. The
cache emits an oauth2client deprecation warning that pollutes the
console; the discovery doc is small, the cache buys nothing here.

## Google Cloud Console state

- Project: `aone-497321`
- Gmail API: Enabled
- OAuth consent screen: **External**, **Testing**, app name `Aone`
- OAuth Client: type **Desktop app**, name `Aone CLI`
- Test users: `axelvil1999@gmail.com`

## Friction during the user-facing flow

Google's consent screen shows a yellow **"Google hasn't verified this
app"** warning while the app is in Testing mode. Users must:

1. Click **Advanced**
2. Click **"Go to Aone (unsafe)"**
3. Then see the actual consent screen

This is fine for v0 (dev + early users we trust). Removing the warning
requires going through Google's app verification (separate workstream,
deferred to v1).

## Next ticket (AONE-202)

Wrap `get_service()` in production-grade code:

- Make `credentials_path` / `token_path` injectable for tests
- Mock the OAuth flow in tests using `unittest.mock`
- Add unit tests for: token reuse, expired-token refresh, missing
  `credentials.json` error path
- Decide whether to expose authentication as part of the CLI
  (e.g. `aone auth login`) or keep it implicit in `aone sync`

## Caveats observed

- The authenticated test account (`axelvil1999@gmail.com`) currently has
  `messagesTotal: 1` in Google's storage. Not a code issue — the
  account is essentially empty. To exercise downstream sprints with
  realistic data, either add another test user or set up a forwarding
  rule before reaching Sprint 3 (storage) or Sprint 4 (agent).
