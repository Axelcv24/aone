# Aone — v0 implementation plan, organized by sprint

Sprint-based board. Structure is still Jira-style (tickets with stable IDs, priority, story points, dependencies, acceptance criteria) but the planning unit is the **sprint**, not the epic.

**Conventions**
- Types: `STORY`, `TASK`, `SPIKE`, `BUG`
- Priority: `P0` (v0 blocker), `P1` (v0 must-have), `P2` (v0 nice-to-have), `P3` (post-v0)
- Status: `Backlog · In Progress · In Review · Done`
- Estimation: Fibonacci (1, 2, 3, 5, 8)
- ID: `AONE-NXX` where `N` = sprint number

**How tickets land in a sprint**
By technical dependency and by "what I can demo at the end". Each sprint closes with a verifiable deliverable, not just code.

---

## 🏁 Executive summary

| Sprint | Theme                                        | Duration | Tickets | Pts | Deliverable |
|--------|----------------------------------------------|----------|---------|-----|-------------|
| S1     | Setup, infra, and workflow                   | 1 week   | 12      | 26  | Public clonable repo, devcontainer boots, CI green, tokens validated |
| S2     | Gmail connector + OAuth                      | 1 week   | 4       | 14  | `uv run python -m aone.gmail.demo` pulls 100 real emails |
| S3     | Storage (cache + FAISS) + LLM wrapper        | 1 week   | 4       | 12  | Cache persists to disk; semantic search returns top-k |
| S4     | LangGraph agent + 5 tools                    | 1.5 wk   | 10      | 30  | `aone ask "question"` returns correct answers locally |
| S5     | CLI + Observability + Evals                  | 1 week   | 9       | 21  | `aone sync/ask/stats/evals` operational; Langfuse traces; RAGAS suite |
| S6     | Docs and v0.1.0 release                      | 3–4 days | 3       | 5   | `v0.1.0` tag, changelog, README with real metrics |
| **Total** |                                           | **5–6 wk** | **42** | **108** | |

---

## Sprint 1 · Setup, infra, and workflow

**Goal**: anyone can clone the repo, open it in VS Code/Cursor with the devcontainer, and have a working environment in under 5 minutes. CI green on `main`. Tokens validated.

### AONE-101 · Create public GitHub repo + MIT LICENSE
- **Type**: TASK · **Priority**: P0 · **Pts**: 1 · **Depends on**: —
- **AC**: public repo with `LICENSE` (MIT), `README.md`, description, and topics (`langgraph`, `agents`, `gmail`, `claude`, `python`).

### AONE-102 · Obtain and validate API tokens (free stack)
- **Type**: TASK · **Priority**: P0 · **Pts**: 2 · **Depends on**: —
- **Description**: Create accounts/projects in:
  - **Groq Console** (https://console.groq.com) → API key. Validate access to `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.
  - **Langfuse Cloud** (https://cloud.langfuse.com) → project + public/secret key + host URL.
  - **Google Cloud Console** → project, enable Gmail API, create OAuth 2.0 Client ID (Desktop app type) → download `credentials.json`.
- **AC**: local `.env` works; smoke test `litellm completion` with `groq/llama-3.3-70b-versatile` responds; `credentials.json` at repo root (gitignored).
- **Optional (post-v0, non-blocking)**: Anthropic / OpenAI / Gemini keys for comparing against paid models.

### AONE-103 · Bootstrap project with `uv`
- **Type**: TASK · **Priority**: P0 · **Pts**: 2 · **Depends on**: AONE-101
- **AC**: `pyproject.toml` with Python 3.12, `src/` layout, `aone` entry point. `uv sync` installs without errors. `uv run aone --help` returns Typer help.

### AONE-104 · Configure base dependencies
- **Type**: TASK · **Priority**: P0 · **Pts**: 2 · **Depends on**: AONE-103
- **Description**: Core: `langgraph`, `litellm`, `faiss-cpu`, `sentence-transformers`, `typer`, `google-api-python-client`, `google-auth-oauthlib`, `langfuse`, `python-dotenv`. Dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`.
- **AC**: `uv lock` reproducible; `uv run pytest` boots; `uv run ruff check src/` passes.

### AONE-105 · Create target directory structure
- **Type**: TASK · **Priority**: P0 · **Pts**: 1 · **Depends on**: AONE-103
- **AC**: `src/aone/{cli,agent,gmail,storage,llm,observability}/__init__.py`, `tests/`, `evals/`, `docs/decisions/`, `.github/` created.

### AONE-106 · `.env.example` and `src/aone/config.py`
- **Type**: TASK · **Priority**: P0 · **Pts**: 2 · **Depends on**: AONE-104
- **Description**: `.env.example` lists every variable; `config.py` loads them with `python-dotenv` and validates presence with clear messages. **Agnostic design**: models come from env, never hardcoded (see ADR-005).
- **Variables**:
  - Required v0: `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
  - Models (with defaults): `AONE_MODEL_GENERATION=groq/llama-3.3-70b-versatile`, `AONE_MODEL_CLASSIFICATION=groq/llama-3.1-8b-instant`
  - Embeddings (with defaults): `AONE_EMBEDDING_PROVIDER=local`, `AONE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
  - Other: `AONE_SYNC_LIMIT=500`
  - Optional (post-v0): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
- **AC**: booting without `.env` fails with a readable error pointing at what's missing; changing `AONE_MODEL_GENERATION` to another provider works as long as that provider's API key is present.

### AONE-107 · Devcontainer + Dockerfile
- **Type**: TASK · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-104
- **Description**: `.devcontainer/devcontainer.json` + `Dockerfile` based on Python 3.12-slim. Post-install runs `uv sync`. Recommended VS Code extensions (`charliermarsh.ruff`, `ms-python.python`).
- **AC**: "Reopen in Container" in VS Code/Cursor boots the env. `uv run aone --help` works inside the container.

### AONE-108 · GitHub Actions CI
- **Type**: TASK · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-104
- **Description**: `.github/workflows/ci.yml` runs on each PR: `uv sync` → `ruff check` → `ruff format --check` → `mypy src/` → `pytest`.
- **AC**: test PR shows green checks; intentional failure breaks the build.

### AONE-109 · Linting + pre-commit setup
- **Type**: TASK · **Priority**: P1 · **Pts**: 2 · **Depends on**: AONE-104
- **Description**: `.pre-commit-config.yaml` with `ruff format`, `ruff check`, `mypy`. `pre-commit install` documented in README.
- **AC**: a local commit with poorly formatted code is auto-formatted.

### AONE-110 · Branch protection + PR/issue templates
- **Type**: TASK · **Priority**: P1 · **Pts**: 2 · **Depends on**: AONE-101, AONE-108
- **Description**: protect `main` (require PR, require status checks, no direct push). `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/{bug,feature}.md`.
- **AC**: direct push to `main` is blocked; opening a PR shows the template pre-loaded.

### AONE-111 · CONTRIBUTING.md + git workflow
- **Type**: TASK · **Priority**: P1 · **Pts**: 2 · **Depends on**: AONE-110
- **Description**: document branching (`feat/`, `fix/`, `chore/`), conventional commits, how to run tests/evals, how to open a PR.
- **AC**: `CONTRIBUTING.md` at repo root covers the 5 scenarios above with copy-paste commands.

### AONE-112 · Write ADRs 001–005
- **Type**: TASK · **Priority**: P1 · **Pts**: 3 · **Depends on**: AONE-105
- **Description**: materialize the ADRs already referenced in `CLAUDE.md` as individual files in `docs/decisions/` with Context / Decision / Consequences format.
- **AC**: `001-no-db-v0.md`, `002-faiss-in-memory.md`, `003-langgraph-over-langchain.md`, `004-mcp-ready-tools.md`, `005-model-agnostic-stack.md` exist and are linked from `README.md`. ADR-005 details: free defaults (Groq + local sentence-transformers), routing via LiteLLM, env vars as the contract for provider switching.

---

## Sprint 2 · Gmail connector + OAuth

**Goal**: pull real Gmail messages to disk. Demo script: `python -m aone.gmail.demo --limit 100`.

### AONE-201 · Spike: Gmail API OAuth desktop flow
- **Type**: SPIKE · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-102, AONE-106
- **AC**: notes in `docs/spikes/gmail-oauth.md` with final scopes (`gmail.readonly`), refresh token handling, `token.json` location.

### AONE-202 · Gmail client: authentication
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-201
- **Description**: `src/aone/gmail/auth.py` with `get_service()` that reuses `token.json` or launches the flow.
- **AC**: works with `credentials.json` at repo root; refreshes expired tokens without intervention; test with a mocked service.

### AONE-203 · Gmail client: list and fetch messages
- **Type**: STORY · **Priority**: P0 · **Pts**: 5 · **Depends on**: AONE-202
- **Description**: `list_messages(limit, query)` and `get_message(id)` returning an `Email` dataclass (id, thread_id, from, to, subject, body_text, body_html, snippet, internal_date, labels).
- **AC**: paginates via `pageToken`; decodes MIME multipart; exponential backoff on 429/5xx; ≥80% coverage in `tests/gmail/`.

### AONE-204 · Message normalizer
- **Type**: TASK · **Priority**: P1 · **Pts**: 3 · **Depends on**: AONE-203
- **Description**: strip signatures, quoted replies, tracking pixels; extract plain text usable for embeddings.
- **AC**: tests with 5 sample emails (signature, reply chain, heavy HTML, newsletter, plain text).

---

## Sprint 3 · Storage + LLM wrapper

**Goal**: idempotent local persistence and working semantic search. LiteLLM routes correctly between Claude/Llama/whatever the env says.

### AONE-301 · `EmailCache` with pickle persistence
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-203
- **Description**: `src/aone/storage/cache.py` — `EmailCache` (dict[id → Email]) + `save()`/`load()` to `~/.aone/cache.pkl`.
- **AC**: idempotent (re-sync doesn't duplicate); atomic `save()` (tmpfile + rename); schema version breaks cleanly on bump.

### AONE-302 · FAISS in-memory index + agnostic embeddings layer
- **Type**: STORY · **Priority**: P0 · **Pts**: 5 · **Depends on**: AONE-301
- **Description**: `src/aone/storage/vector.py` wrapping FAISS `IndexFlatL2`. **Agnostic embeddings layer** (`src/aone/llm/embeddings.py`) that supports two providers based on `AONE_EMBEDDING_PROVIDER`:
  - `local` (v0 default) → `sentence-transformers/all-MiniLM-L6-v2` (384 dims, CPU, free)
  - `litellm` → any model LiteLLM supports (OpenAI, Voyage, Cohere, etc.)
  
  Persists to `~/.aone/index.faiss` + `meta.json` (pos↔id mapping + dims + provider used).
- **AC**: `add(email)` and `search(query, k)` work with both providers; reload preserves order; similarity > 0.7 on a relevant sample; switching from `local` to `litellm` requires re-indexing (validate detection + warn the user).

### AONE-303 · Cache statistics
- **Type**: TASK · **Priority**: P2 · **Pts**: 1 · **Depends on**: AONE-301, AONE-302
- **Description**: `stats()` with email count, min/max dates, top 5 senders, on-disk size.

### AONE-304 · Provider-agnostic LiteLLM wrapper
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-106
- **Description**: `src/aone/llm/client.py` — `complete(messages, model=None, role=None, **kwargs)`. When `model` is `None`, the wrapper reads `AONE_MODEL_GENERATION` or maps from `role` (`"generation"` / `"classification"`). LiteLLM already supports Groq, Anthropic, OpenAI, Gemini, etc. with the same signature — the wrapper adds config reads, retries, and Langfuse instrumentation.
- **AC**: streaming + non-streaming; retry with backoff; switching `AONE_MODEL_GENERATION` from `groq/llama-3.3-70b-versatile` to `anthropic/claude-haiku-4-5` works without touching code as long as the matching API key is set; tests with recorded responses for Groq + a mock alternative provider.

---

## Sprint 4 · LangGraph agent + tools

**Goal**: the graph answers real questions about the cached emails end-to-end. Callable from Python (the CLI wire-up lands in S5).

### AONE-401 · `classify_intent` node
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-304
- **Description**: uses the wrapper with `role="classification"` → resolves to `AONE_MODEL_CLASSIFICATION` (default `groq/llama-3.1-8b-instant`). Intents: `summarize`, `find_emails`, `aggregate_amounts`, `list_contacts`, `general_qa`.
- **AC**: ≥85% accuracy on 20 seeded examples with the default model; switching to another model in `.env` does not require editing this node.

### AONE-402 · `select_tools` node
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-401
- **Description**: maps intent → list of tools. Deterministic logic (no LLM).

### AONE-403 · `search_emails` tool
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-302, AONE-402
- **Description**: semantic search over FAISS + filters (sender, date_range, label). MCP-compatible signature (JSON schema).

### AONE-404 · `get_thread` tool
- **Type**: STORY · **Priority**: P0 · **Pts**: 2 · **Depends on**: AONE-301, AONE-402

### AONE-405 · `list_contacts` tool
- **Type**: STORY · **Priority**: P1 · **Pts**: 2 · **Depends on**: AONE-301, AONE-402
- **Description**: returns unique contacts with message count and last seen date.

### AONE-406 · `aggregate_amounts` tool
- **Type**: STORY · **Priority**: P1 · **Pts**: 5 · **Depends on**: AONE-403
- **Description**: extracts monetary amounts and sums them per contact/date. Regex + validation via the LLM wrapper (`role="generation"`).

### AONE-407 · `summarize_thread` tool
- **Type**: STORY · **Priority**: P1 · **Pts**: 3 · **Depends on**: AONE-404
- **Description**: summarizes a thread or set of emails via the LLM wrapper (`role="generation"`, default `groq/llama-3.3-70b-versatile`).

### AONE-408 · `execute_tools` node
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-403, AONE-404, AONE-405, AONE-406, AONE-407
- **Description**: runs the selected tools in parallel when they're independent; accumulates results in the graph state.

### AONE-409 · `generate_response` node
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-408
- **Description**: produces the final answer via the LLM wrapper (`role="generation"`, default Llama 3.3 70B via Groq) consuming the accumulated state.
- **AC**: answers cite emails by subject/date when applicable; switching `AONE_MODEL_GENERATION` does not break the node.

### AONE-410 · Assemble the LangGraph graph
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-401, AONE-402, AONE-408, AONE-409
- **Description**: `src/aone/agent/graph.py` builds the `StateGraph` with the four nodes wired up.

---

## Sprint 5 · CLI + Observability + Evals

**Goal**: usable from the terminal with full tracing in Langfuse and an eval suite that runs locally + in CI.

### AONE-501 · `aone sync`
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-203, AONE-301, AONE-302
- **Description**: syncs the last N emails (default 500, `--limit` override). Progress bar.

### AONE-502 · `aone ask`
- **Type**: STORY · **Priority**: P0 · **Pts**: 3 · **Depends on**: AONE-410

### AONE-503 · `aone stats`
- **Type**: TASK · **Priority**: P2 · **Pts**: 1 · **Depends on**: AONE-303

### AONE-504 · `aone evals`
- **Type**: TASK · **Priority**: P1 · **Pts**: 2 · **Depends on**: AONE-509

### AONE-505 · Langfuse SDK integration
- **Type**: STORY · **Priority**: P1 · **Pts**: 3 · **Depends on**: AONE-304
- **Description**: decorate graph nodes and tools with `@observe()`. Capture input/output and costs.
- **AC**: one `aone ask` run shows up as a full trace in the dashboard.

### AONE-506 · Tags and metadata on traces
- **Type**: TASK · **Priority**: P2 · **Pts**: 1 · **Depends on**: AONE-505
- **Description**: tag by intent, session, agent version. Useful to filter evals vs production.

### AONE-507 · `golden_set.jsonl` with 20+ questions
- **Type**: TASK · **Priority**: P1 · **Pts**: 3 · **Depends on**: AONE-105
- **Description**: schema: `{question, expected_intent, expected_keywords, expected_emails_referenced, notes}`.

### AONE-508 · RAGAS runner
- **Type**: STORY · **Priority**: P1 · **Pts**: 5 · **Depends on**: AONE-410, AONE-507
- **Description**: metrics: faithfulness, answer_relevancy, context_precision. Prints a stdout table + JSON to `evals/results/`.

### AONE-509 · Regression threshold
- **Type**: TASK · **Priority**: P2 · **Pts**: 2 · **Depends on**: AONE-508
- **Description**: `aone evals --fail-under 0.7` returns a non-zero exit code if the average metric drops below the threshold. Wired into CI as an opt-in manual check.

---

## Sprint 6 · Docs and v0.1.0 release

**Goal**: repo ready to share/showcase. Real metrics in README.

### AONE-601 · `docs/architecture.md`
- **Type**: TASK · **Priority**: P1 · **Pts**: 3 · **Depends on**: AONE-410
- **Description**: final graph diagram, data lifecycle, design decisions linked to ADRs.

### AONE-602 · Update README with real metrics
- **Type**: TASK · **Priority**: P1 · **Pts**: 1 · **Depends on**: AONE-508
- **Description**: fill in the "Metrics (v0.1.0)" section: golden set accuracy, p95 latency, average cost per query.

### AONE-603 · `v0.1.0` tag + changelog
- **Type**: TASK · **Priority**: P1 · **Pts**: 1 · **Depends on**: AONE-501, AONE-502, AONE-505, AONE-508
- **AC**: `CHANGELOG.md` created, annotated git tag, release notes mention known limitations.

---

## Post-v0 backlog (reference, not prioritized)

- AONE-901 · PostgreSQL + pgvector migration
- AONE-902 · REST API with FastAPI
- AONE-903 · Next.js frontend
- AONE-904 · Multi-user auth
- AONE-905 · PDF attachment processing
- AONE-906 · Outlook connector
- AONE-907 · Slack connector
- AONE-908 · Expose tools as an MCP server
