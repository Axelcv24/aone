# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Aone is a conversational operations agent for freelancers and SMBs. It ingests Gmail messages and answers business questions in natural language from the terminal.

## Repository status

**Pre-implementation**: as of this writing the repo only contains `README.md` and `CLAUDE.md`. The directory tree below is the *target* layout — none of it exists on disk yet. When starting work, scaffold what you need rather than assuming files are present.

## Stack and key versions

- Python 3.12 — all dependencies managed via `uv` (never pip/poetry)
- LangGraph for agent orchestration (state machine, not flat LangChain)
- **LiteLLM** as the model wrapper. v0 defaults to 100% free providers; all model IDs come from env vars so swapping to paid models (Claude, GPT-4o, etc.) is a `.env` change, not a code change. See ADR-005.
  - Generation (default): `groq/llama-3.3-70b-versatile` (Groq free tier)
  - Classification (default): `groq/llama-3.1-8b-instant` (Groq free tier)
- **Embeddings** via local `sentence-transformers/all-MiniLM-L6-v2` by default (free, runs on CPU). Switchable to LiteLLM-routed embeddings (OpenAI/Voyage/etc.) via `AONE_EMBEDDING_PROVIDER=litellm`.
- FAISS in-memory vector index (no Pinecone/Qdrant in v0 — see ADR-002)
- Gmail API via OAuth desktop flow (requires `credentials.json` in repo root)
- Langfuse Cloud free tier for tracing/observability (50k events/mo). Self-hosted is an option if we outgrow it.
- Typer for the CLI entrypoint

Required environment (free-tier defaults):
- `GROQ_API_KEY` — generation + classification
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` — tracing
- Gmail OAuth `credentials.json` in the project root

Optional (only if swapping to paid models): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`. LiteLLM reads whichever is present based on the model ID configured.

## Architecture

Single pipeline, four LangGraph nodes. Models referenced by env var, not hardcoded:

```
CLI (Typer)
  → classify_intent      (AONE_MODEL_CLASSIFICATION, default groq/llama-3.1-8b-instant)
  → select_tools
  → execute_tools        ← MCP-compatible tool surface (ADR-004)
  → generate_response    (AONE_MODEL_GENERATION, default groq/llama-3.3-70b-versatile)
```

Data layer is deliberately **persistence-free** in v0 (ADR-001): an in-memory `EmailCache` dict plus a FAISS index, both serialized to pickle/JSON between runs. Postgres + pgvector migration is reserved for v1 and is documented in `docs/decisions/`.

Tools should be authored to be MCP-compatible from day one (ADR-004) so they can later be exposed via MCP servers in v2 without rewriting.

## Target layout

```
src/
  cli.py              # Typer entrypoint
  agent/              # LangGraph graph + tool implementations
  gmail/              # Gmail API client
  storage/            # EmailCache + FAISS index, pickle/JSON I/O
  llm/                # LiteLLM wrapper (model routing lives here)
  observability/      # Langfuse client/decorators
tests/
evals/golden_set.jsonl   # 20+ Q&A with ground truth — drives development
docs/
  architecture.md
  decisions/          # ADRs 001–004 (see below)
```

## CLI surface (planned)

- `uv run aone sync` — pull last N Gmail messages into the cache
- `uv run aone ask "<question>"` — query the agent
- `uv run aone stats` — local cache statistics
- `uv run aone evals` — run the eval suite

Tests run with pytest (`uv run pytest`); use `uv run pytest tests/path/to/test_file.py::test_name` for a single test.

## Active ADRs

- **001** — No DB in v0; memory + pickle only
- **002** — FAISS in-memory (not Pinecone/Qdrant) for v0
- **003** — LangGraph over flat LangChain
- **004** — MCP-ready tool surface from the start
- **005** — Model-agnostic stack: providers/models come from env vars, LiteLLM routes them. v0 defaults are 100% free (Groq + local sentence-transformers); paid models (Claude, GPT, Gemini) are a `.env` swap.

When making architectural choices that touch any of these, update the ADR rather than silently diverging.

## Working conventions

- **Eval-driven development** (Notion AI style): when changing agent behavior, update or add cases in `evals/golden_set.jsonl` first, then iterate on tool/prompt changes until the suite passes. Don't ship agent changes without an eval that locks in the new behavior.
- **Five-component agent pattern** (Vercel style): keep the classify → select → execute → respond split clean; resist collapsing nodes for convenience.
- **Roadmap discipline**: v0 is validation-of-concept. Don't preemptively pull in FastAPI, a frontend, multi-tenant auth, or extra connectors (Outlook/Slack) — those are explicitly v1/v2.

## Language note

README and design docs are in Spanish; code identifiers and commit messages should follow standard English conventions.
