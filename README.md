# Aone

> Agente conversacional de operaciones para freelancers y PYMES. Procesa tus correos de Gmail y responde preguntas de negocio en lenguaje natural directamente desde la terminal.

## Qué hace

```bash
$ aone ask "¿cuánto me debe Acme?"
Acme te debe $3,450 USD distribuidos en 3 facturas:
  • Factura #1024 ($1,200) vencida hace 15 días
  • Factura #1031 ($1,500) vence en 5 días
  • Factura #1042 ($750) vence en 20 días

$ aone ask "resume mis conversaciones con Acme este mes"
[resumen generado por Claude basado en tus correos reales]

$ aone ask "¿qué clientes no me han respondido en más de 30 días?"
[lista priorizada de clientes inactivos]
```

## Stack

- **Python 3.12** + **uv** para gestión de dependencias
- **LangGraph** para orquestación del agente
- **Claude Sonnet 4.6** (Anthropic) como modelo principal
- **GPT-4o-mini** para clasificación de intención (más barato)
- **FAISS** para búsqueda semántica in-memory
- **Gmail API** para acceso a correos
- **Langfuse** para tracing y observabilidad

## Arquitectura

```
┌──────────────────────────────────────────────┐
│  CLI (Typer)                                 │
└────────────────────┬─────────────────────────┘
                     ↓
┌──────────────────────────────────────────────┐
│  Agente (LangGraph state machine)            │
│    classify_intent → select_tools →          │
│    execute_tools → generate_response         │
└────────────────────┬─────────────────────────┘
                     ↓
┌──────────────────────────────────────────────┐
│  Capa de datos (in-memory + pickle)          │
│    EmailCache (dict) + FAISS index           │
└─────────────────────┬────────────────────────┘
                     ↓ sync periódico
┌──────────────────────────────────────────────┐
│  Gmail API (OAuth desktop)                   │
└──────────────────────────────────────────────┘
```

Diseño explícitamente sin base de datos en v0. Toda la persistencia es local (pickle + JSON). Path de migración a Postgres documentado en `docs/decisions/`.

## Quickstart

### Pre-requisitos
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) instalado
- API keys de [Anthropic](https://console.anthropic.com) y [OpenAI](https://platform.openai.com)
- OAuth credentials de [Google Cloud Console](https://console.cloud.google.com) (Gmail API habilitada)

### Setup

```bash
# Clonar y entrar
git clone https://github.com/[tu-usuario]/aone.git
cd aone

# Instalar dependencias
uv sync

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# Poner credentials.json de Google en la raíz del proyecto

# Sincronizar correos por primera vez (abre navegador para OAuth)
uv run aone sync

# Hacer tu primera pregunta
uv run aone ask "¿cuántos correos tengo de la última semana?"
```

### Comandos disponibles

```bash
uv run aone sync              # sincroniza últimos N correos de Gmail
uv run aone ask "pregunta"    # consulta al agente
uv run aone stats             # estadísticas del cache local
uv run aone evals             # corre suite de evaluaciones
```

## Estructura del proyecto

```
aone/
├── src/
│   ├── cli.py                 # entrypoint Typer
│   ├── agent/                 # LangGraph + tools
│   ├── gmail/                 # cliente Gmail API
│   ├── storage/               # cache + FAISS
│   ├── llm/                   # wrapper LiteLLM
│   └── observability/         # Langfuse
├── tests/
├── evals/
│   └── golden_set.jsonl       # 20+ preguntas con ground truth
└── docs/
    ├── architecture.md
    └── decisions/             # ADRs
```

## Decisiones técnicas

Documentadas como ADRs en `docs/decisions/`:

- **001**: Sin DB en v0, solo memoria + pickle
- **002**: FAISS in-memory sobre Pinecone/Qdrant para v0
- **003**: LangGraph sobre LangChain plano
- **004**: MCP-ready architecture para tools

## Métricas (v0.1.0)

> Se actualizan a medida que se completa el desarrollo

- Tasa de aciertos en golden set: TBD
- Latencia p95: TBD
- Costo promedio por query: TBD

## Roadmap

### v0 (actual) — Validación de concepto
- [x] Setup inicial
- [ ] Gmail connector + OAuth
- [ ] Cache local con pickle
- [ ] FAISS in-memory para búsqueda semántica
- [ ] LangGraph state machine con 5 tools
- [ ] Suite de evals con RAGAS
- [ ] Langfuse tracing

### v1 — Persistencia y UI
- [ ] Migración a PostgreSQL + pgvector
- [ ] API REST con FastAPI
- [ ] Frontend Next.js
- [ ] Auth multi-usuario
- [ ] Procesamiento de facturas PDF
- [ ] Deploy en cloud

### v2 — Multi-vertical
- [ ] Conectores adicionales (Outlook, Slack)
- [ ] Templates para verticales (contadores, abogados, agencias)
- [ ] MCP servers exponibles a clientes externos
- [ ] SaaS multi-tenant

## Inspiración técnica

Construido siguiendo prácticas de:
- Anthropic — "Building Effective Agents"
- Klarna — uso de LangGraph en producción
- Notion AI — eval-driven development
- Vercel — patrón de 5 componentes para agentes

## Licencia

MIT — ver [LICENSE](./LICENSE)

## Autor

[Tu nombre] · [LinkedIn] · [Twitter/X]

---

**Status**: 🚧 En desarrollo activo · v0.1.0
