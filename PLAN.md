# Aone — Plan de implementación v0 por sprints

Tablero por sprints. La estructura sigue siendo estilo Jira (tickets con ID estable, prioridad, story points, dependencias, criterios de aceptación) pero ahora la unidad de planificación es el **sprint**, no la épica.

**Convenciones**
- Tipos: `STORY`, `TASK`, `SPIKE`, `BUG`
- Prioridad: `P0` (bloqueante v0), `P1` (necesario v0), `P2` (deseable v0), `P3` (post-v0)
- Estado: `Backlog · In Progress · In Review · Done`
- Estimación: Fibonacci (1, 2, 3, 5, 8)
- ID: `AONE-NXX` donde `N` = número de sprint

**Cómo se decide qué entra en cada sprint**
Por dependencia técnica y por nivel de "puedo demostrar algo": al cierre de cada sprint hay un entregable verificable, no solo código.

---

## 🏁 Resumen ejecutivo

| Sprint | Tema                                          | Duración | Tickets | Pts | Entregable |
|--------|-----------------------------------------------|----------|---------|-----|------------|
| S1     | Setup, infra y workflow                       | 1 sem    | 12      | 25  | Repo público clonable, devcontainer arranca, CI verde, tokens validados |
| S2     | Conector Gmail + OAuth                        | 1 sem    | 4       | 14  | `uv run python -m aone.gmail.demo` baja 100 correos reales |
| S3     | Almacenamiento (cache + FAISS) + LLM wrapper  | 1 sem    | 4       | 12  | Cache persiste a disco; búsqueda semántica devuelve top-k |
| S4     | Agente LangGraph + 5 tools                    | 1.5 sem  | 10      | 30  | `aone ask "pregunta"` responde correctamente en local |
| S5     | CLI + Observabilidad + Evals                  | 1 sem    | 9       | 21  | `aone sync/ask/stats/evals` operativos; traces en Langfuse; suite RAGAS |
| S6     | Docs y release v0.1.0                         | 3-4 días | 3       | 5   | Tag `v0.1.0`, changelog, README con métricas reales |
| **Total** |                                            | **5-6 sem** | **42** | **107** | |

---

## Sprint 1 · Setup, infra y workflow

**Goal**: cualquier persona puede clonar el repo, abrirlo en VS Code/Cursor con devcontainer, y tener un entorno funcional en <5 min. CI verde en `main`. Tokens validados.

### AONE-101 · Crear repo público en GitHub + LICENSE MIT
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 1 · **Depende de**: —
- **AC**: repo público con `LICENSE` (MIT), `README.md` movido, descripción y topics (`langgraph`, `agents`, `gmail`, `claude`, `python`).

### AONE-102 · Conseguir y validar API tokens (stack gratis)
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 2 · **Depende de**: —
- **Descripción**: Crear cuentas/proyectos:
  - **Groq Console** (https://console.groq.com) → API key. Validar acceso a `llama-3.3-70b-versatile` y `llama-3.1-8b-instant`.
  - **Langfuse Cloud** (https://cloud.langfuse.com) → proyecto + public/secret key + host URL.
  - **Google Cloud Console** → proyecto, habilitar Gmail API, crear OAuth 2.0 Client ID tipo *Desktop app* → descargar `credentials.json`.
- **AC**: `.env` local funciona; smoke test `litellm completion` con `groq/llama-3.3-70b-versatile` responde; `credentials.json` en raíz (gitignored).
- **Opcional (post-v0, no bloquea)**: Anthropic / OpenAI / Gemini keys si se quiere comparar contra modelos pagos.

### AONE-103 · Inicializar proyecto con `uv`
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 2 · **Depende de**: AONE-101
- **AC**: `pyproject.toml` con Python 3.12, layout `src/`, entry point `aone`. `uv sync` instala sin errores. `uv run aone --help` devuelve ayuda de Typer.

### AONE-104 · Configurar dependencias base
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 2 · **Depende de**: AONE-103
- **Descripción**: Core: `langgraph`, `litellm`, `faiss-cpu`, `sentence-transformers`, `typer`, `google-api-python-client`, `google-auth-oauthlib`, `langfuse`, `python-dotenv`. Dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`.
- **AC**: `uv lock` reproducible; `uv run pytest` arranca; `uv run ruff check src/` pasa.

### AONE-105 · Crear estructura de directorios objetivo
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 1 · **Depende de**: AONE-103
- **AC**: `src/{cli,agent,gmail,storage,llm,observability}/__init__.py`, `tests/`, `evals/`, `docs/decisions/`, `.github/` creados.

### AONE-106 · `.env.example` y `src/config.py`
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 2 · **Depende de**: AONE-104
- **Descripción**: `.env.example` lista todas las vars; `config.py` las carga con `python-dotenv` y valida presencia con mensajes claros. **Diseño agnóstico**: los modelos vienen de env, no hardcodeados (ver ADR-005).
- **Variables**:
  - Requeridas v0: `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
  - Modelos (con defaults): `AONE_MODEL_GENERATION=groq/llama-3.3-70b-versatile`, `AONE_MODEL_CLASSIFICATION=groq/llama-3.1-8b-instant`
  - Embeddings (con defaults): `AONE_EMBEDDING_PROVIDER=local`, `AONE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
  - Otros: `AONE_SYNC_LIMIT=500`
  - Opcionales (post-v0): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
- **AC**: arrancar sin `.env` falla con error legible que dice qué falta; cambiar `AONE_MODEL_GENERATION` a otro provider funciona si su API key está presente.

### AONE-107 · Devcontainer + Dockerfile
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-104
- **Descripción**: `.devcontainer/devcontainer.json` + `Dockerfile` base Python 3.12-slim. Postinstall corre `uv sync`. Extensiones VS Code recomendadas (`charliermarsh.ruff`, `ms-python.python`).
- **AC**: "Reopen in Container" en VS Code/Cursor levanta el entorno. `uv run aone --help` funciona dentro del contenedor.

### AONE-108 · GitHub Actions CI
- **Tipo**: TASK · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-104
- **Descripción**: Workflow `.github/workflows/ci.yml` que en cada PR corre: `uv sync` → `ruff check` → `ruff format --check` → `mypy src/` → `pytest`.
- **AC**: PR de prueba muestra checks verdes; falla intencional rompe el build.

### AONE-109 · Setup linting + pre-commit
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 2 · **Depende de**: AONE-104
- **Descripción**: `.pre-commit-config.yaml` con `ruff format`, `ruff check`, `mypy`. Hook `pre-commit install` documentado en README.
- **AC**: commit local con código mal formateado se autoformatea.

### AONE-110 · Branch protection + PR/issue templates
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 2 · **Depende de**: AONE-101, AONE-108
- **Descripción**: Proteger `main` (require PR, require status checks, no direct push). `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/{bug,feature}.md`.
- **AC**: no se puede pushear directo a `main`; abrir PR muestra template precargado.

### AONE-111 · CONTRIBUTING.md + git workflow
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 2 · **Depende de**: AONE-110
- **Descripción**: Documentar branching (`feat/`, `fix/`, `chore/`), conventional commits, cómo correr tests/evals, cómo abrir PR.
- **AC**: `CONTRIBUTING.md` en raíz cubre los 5 escenarios anteriores con comandos copiables.

### AONE-112 · Escribir los ADRs 001-005
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 3 · **Depende de**: AONE-105
- **Descripción**: Materializar los ADRs ya referenciados en `CLAUDE.md` como archivos individuales en `docs/decisions/` con formato Context / Decision / Consequences.
- **AC**: `001-no-db-v0.md`, `002-faiss-in-memory.md`, `003-langgraph-over-langchain.md`, `004-mcp-ready-tools.md`, `005-model-agnostic-stack.md` existen y están enlazados desde `README.md`. ADR-005 detalla: defaults free (Groq + sentence-transformers locales), routing via LiteLLM, env vars como contrato.

---

## Sprint 2 · Conector Gmail + OAuth

**Goal**: bajar correos reales de Gmail al disco. Demo script: `python -m aone.gmail.demo --limit 100`.

### AONE-201 · Spike: flujo OAuth desktop con Gmail API
- **Tipo**: SPIKE · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-102, AONE-106
- **AC**: notas en `docs/spikes/gmail-oauth.md` con scopes finales (`gmail.readonly`), manejo de refresh tokens, ubicación de `token.json`.

### AONE-202 · Cliente Gmail: autenticación
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-201
- **Descripción**: `src/gmail/auth.py` con `get_service()` que reusa `token.json` o lanza el flow.
- **AC**: funciona con `credentials.json` en raíz; refresca token expirado sin intervención; test con `Mock(build)`.

### AONE-203 · Cliente Gmail: listar y descargar mensajes
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 5 · **Depende de**: AONE-202
- **Descripción**: `list_messages(limit, query)` y `get_message(id)` devolviendo dataclass `Email` (id, thread_id, from, to, subject, body_text, body_html, snippet, internal_date, labels).
- **AC**: pagina con `pageToken`; decodifica MIME multipart; backoff exponencial en 429/5xx; cobertura ≥80% en `tests/gmail/`.

### AONE-204 · Normalizador de mensajes
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 3 · **Depende de**: AONE-203
- **Descripción**: Limpiar firmas, quoted replies, tracking pixels; extraer texto plano útil para embeddings.
- **AC**: tests con 5 correos de muestra (firma, reply chain, HTML pesado, newsletter, plain text).

---

## Sprint 3 · Storage + LLM wrapper

**Goal**: persistencia local idempotente y búsqueda semántica funcional. LiteLLM rutea correctamente entre Claude y GPT-4o-mini.

### AONE-301 · `EmailCache` con persistencia pickle
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-203
- **Descripción**: `src/storage/cache.py` — `EmailCache` (dict[id → Email]) + `save()`/`load()` sobre `~/.aone/cache.pkl`.
- **AC**: idempotente (re-sync no duplica); `save()` atómico (tmpfile + rename); versionado de schema rompe limpio si cambia.

### AONE-302 · Índice FAISS in-memory + capa de embeddings agnóstica
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 5 · **Depende de**: AONE-301
- **Descripción**: `src/storage/vector.py` envolviendo FAISS `IndexFlatL2`. **Capa de embeddings agnóstica** (`src/llm/embeddings.py`) que soporta dos providers según `AONE_EMBEDDING_PROVIDER`:
  - `local` (default v0) → `sentence-transformers/all-MiniLM-L6-v2` (384 dims, CPU, gratis)
  - `litellm` → cualquier modelo soportado por LiteLLM (OpenAI, Voyage, Cohere, etc.)
  
  Persiste a `~/.aone/index.faiss` + `meta.json` (mapping pos↔id + dims + provider usado).
- **AC**: `add(email)` y `search(query, k)` funcionan con ambos providers; recarga preserva orden; similitud > 0.7 en sample relevante; cambiar de `local` a `litellm` requiere re-indexar (validar que se detecte y avise).

### AONE-303 · Estadísticas del cache
- **Tipo**: TASK · **Prioridad**: P2 · **Pts**: 1 · **Depende de**: AONE-301, AONE-302
- **Descripción**: `stats()` con: nº correos, fecha más antigua/reciente, top 5 remitentes, tamaño en disco.

### AONE-304 · Wrapper LiteLLM agnóstico al proveedor
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-106
- **Descripción**: `src/llm/client.py` — `complete(messages, model=None, **kwargs)`. Si `model=None` lo lee de `AONE_MODEL_GENERATION` (o de un parámetro `role` que mapee a `generation`/`classification`). LiteLLM ya soporta Groq, Anthropic, OpenAI, Gemini, etc. con la misma firma — el wrapper solo añade: lectura de config, retry, instrumentación Langfuse.
- **AC**: streaming + non-streaming; retry con backoff; cambiar `AONE_MODEL_GENERATION` de `groq/llama-3.3-70b-versatile` a `anthropic/claude-haiku-4-5` funciona sin tocar código si la API key correspondiente está presente; tests con respuestas grabadas para Groq + un provider alternativo (mock).

---

## Sprint 4 · Agente LangGraph + tools

**Goal**: el grafo end-to-end responde preguntas reales sobre los correos en cache. Llamable desde Python (la CLI viene en S5).

### AONE-401 · Nodo `classify_intent`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-304
- **Descripción**: Usa el wrapper con `role="classification"` → resuelve a `AONE_MODEL_CLASSIFICATION` (default `groq/llama-3.1-8b-instant`). Intents: `summarize`, `find_emails`, `aggregate_amounts`, `list_contacts`, `general_qa`.
- **AC**: ≥85% accuracy en 20 ejemplos sembrados con el modelo default; cambiar a otro modelo en `.env` no requiere editar este nodo.

### AONE-402 · Nodo `select_tools`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-401
- **Descripción**: Mapea intent → lista de tools. Lógica determinística (no LLM).

### AONE-403 · Tool `search_emails`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-302, AONE-402
- **Descripción**: Búsqueda semántica sobre FAISS + filtros (sender, date_range, label). Firma MCP-compatible (JSON schema).

### AONE-404 · Tool `get_thread`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 2 · **Depende de**: AONE-301, AONE-402

### AONE-405 · Tool `list_contacts`
- **Tipo**: STORY · **Prioridad**: P1 · **Pts**: 2 · **Depende de**: AONE-301, AONE-402

### AONE-406 · Tool `aggregate_amounts`
- **Tipo**: STORY · **Prioridad**: P1 · **Pts**: 5 · **Depende de**: AONE-403
- **Descripción**: Extrae montos monetarios y los suma por contacto/fecha. Regex + validación via wrapper LLM (`role="generation"`).

### AONE-407 · Tool `summarize_thread`
- **Tipo**: STORY · **Prioridad**: P1 · **Pts**: 3 · **Depende de**: AONE-404
- **Descripción**: Resume hilo o conjunto de correos via wrapper LLM (`role="generation"`, default `groq/llama-3.3-70b-versatile`).

### AONE-408 · Nodo `execute_tools`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-403, AONE-404, AONE-405, AONE-406, AONE-407
- **Descripción**: Ejecuta tools en paralelo cuando son independientes; acumula resultados en el estado del grafo.

### AONE-409 · Nodo `generate_response`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-408
- **Descripción**: Genera respuesta final via wrapper LLM (`role="generation"`, default Llama 3.3 70B via Groq) consumiendo el estado acumulado.
- **AC**: respuestas citan correos por subject/fecha cuando aplica; cambiar `AONE_MODEL_GENERATION` no rompe el nodo.

### AONE-410 · Ensamblar el grafo LangGraph
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-401, AONE-402, AONE-408, AONE-409
- **Descripción**: `src/agent/graph.py` construye el `StateGraph` con los 4 nodos y conexiones.

---

## Sprint 5 · CLI + Observabilidad + Evals

**Goal**: producto utilizable desde terminal con tracing completo en Langfuse y suite de evals que corre en local + CI.

### AONE-501 · `aone sync`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-203, AONE-301, AONE-302
- **Descripción**: Sincroniza últimos N correos (default 500, override `--limit`). Progress bar.

### AONE-502 · `aone ask`
- **Tipo**: STORY · **Prioridad**: P0 · **Pts**: 3 · **Depende de**: AONE-410

### AONE-503 · `aone stats`
- **Tipo**: TASK · **Prioridad**: P2 · **Pts**: 1 · **Depende de**: AONE-303

### AONE-504 · `aone evals`
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 2 · **Depende de**: AONE-509

### AONE-505 · Integrar Langfuse SDK
- **Tipo**: STORY · **Prioridad**: P1 · **Pts**: 3 · **Depende de**: AONE-304
- **Descripción**: Decorar nodos y tools con `@observe()`. Capturar input/output y costos.
- **AC**: una corrida de `aone ask` aparece como trace completa en dashboard.

### AONE-506 · Tags y metadata en traces
- **Tipo**: TASK · **Prioridad**: P2 · **Pts**: 1 · **Depende de**: AONE-505
- **Descripción**: Tag por intent, sesión, versión del agente.

### AONE-507 · `golden_set.jsonl` con 20+ preguntas
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 3 · **Depende de**: AONE-105
- **Descripción**: Formato: `{question, expected_intent, expected_keywords, expected_emails_referenced, notes}`.

### AONE-508 · Runner RAGAS
- **Tipo**: STORY · **Prioridad**: P1 · **Pts**: 5 · **Depende de**: AONE-410, AONE-507
- **Descripción**: Métricas: faithfulness, answer_relevancy, context_precision. Reporta tabla stdout + JSON en `evals/results/`.

### AONE-509 · Threshold de regresión
- **Tipo**: TASK · **Prioridad**: P2 · **Pts**: 2 · **Depende de**: AONE-508
- **Descripción**: `aone evals --fail-under 0.7` retorna exit code ≠0. Integrar en CI como check opcional manual.

---

## Sprint 6 · Docs y release v0.1.0

**Goal**: repo listo para mostrar/compartir. Métricas reales en README.

### AONE-601 · `docs/architecture.md`
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 3 · **Depende de**: AONE-410
- **Descripción**: Diagrama final del grafo, ciclo de vida de datos, decisiones de diseño con links a ADRs.

### AONE-602 · Actualizar README con métricas reales
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 1 · **Depende de**: AONE-508
- **Descripción**: Rellenar sección "Métricas (v0.1.0)" con: accuracy golden set, latencia p95, costo promedio por query.

### AONE-603 · Tag `v0.1.0` y changelog
- **Tipo**: TASK · **Prioridad**: P1 · **Pts**: 1 · **Depende de**: AONE-501, AONE-502, AONE-505, AONE-508
- **AC**: `CHANGELOG.md` creado, tag git anotado, release notes mencionan limitaciones conocidas.

---

## Backlog post-v0 (referencia)

- AONE-901 · Migración a PostgreSQL + pgvector
- AONE-902 · API REST con FastAPI
- AONE-903 · Frontend Next.js
- AONE-904 · Auth multi-usuario
- AONE-905 · Procesamiento de PDFs adjuntos
- AONE-906 · Conector Outlook
- AONE-907 · Conector Slack
- AONE-908 · Exposición de tools como MCP server
