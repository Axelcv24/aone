#!/usr/bin/env python3
"""Hello World: Aone answering business questions about a synthetic inbox.

The fake invoice emails below stand in for real Gmail messages. Everything
else is production code — EmailCache, VectorIndex, LocalEmbedder,
LLMClient — so the answer comes from the same pipeline ``aone ask`` will
use once Sprint 4/5 wires it up end-to-end.

Run the default battery of questions::

    uv run python examples/ask_about_invoices.py

Or ask anything you want::

    uv run python examples/ask_about_invoices.py "¿Cuál es el saldo pendiente más grande?"
    uv run python examples/ask_about_invoices.py "When is invoice 1031 due?"

To test against your real Gmail instead, replace ``SAMPLE_INVOICES`` with a
list of ``Email`` instances pulled via ``aone.gmail.client.get_message``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from aone.gmail.normalize import normalize
from aone.gmail.types import Email
from aone.llm.client import LLMClient
from aone.llm.embeddings import LocalEmbedder
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex


# ─── Edit these to test different scenarios ──────────────────────────


@dataclass
class InvoiceSample:
    id: str
    from_: str
    to: str
    subject: str
    date_ms: int
    body: str


SAMPLE_INVOICES: list[InvoiceSample] = [
    InvoiceSample(
        id="msg-acme-1024",
        from_="Acme Corp Billing <billing@acme.com>",
        to="axel@example.com",
        subject="Invoice #1024 — REMINDER (15 days overdue)",
        date_ms=1715961600000,
        body="""Hi Axel,

This is a friendly reminder that invoice #1024 for $1,200.00 USD is now 15
days past due.

Invoice details:
  • Invoice number: 1024
  • Original due date: 2026-05-09
  • Amount due: $1,200.00 USD
  • Status: OVERDUE

Please settle at your earliest convenience to avoid late fees.

Best,
Acme Billing Team
""",
    ),
    InvoiceSample(
        id="msg-acme-1031",
        from_="Acme Corp Billing <billing@acme.com>",
        to="axel@example.com",
        subject="Invoice #1031 — Due 2026-05-29",
        date_ms=1716480000000,
        body="""Hi Axel,

Please find invoice #1031 below.

  • Invoice number: 1031
  • Amount: $1,500.00 USD
  • Due date: 2026-05-29 (5 days)
  • Status: PENDING

Acme Corp Billing Team
""",
    ),
    InvoiceSample(
        id="msg-acme-1042",
        from_="Acme Corp Billing <billing@acme.com>",
        to="axel@example.com",
        subject="Invoice #1042 — Due 2026-06-13",
        date_ms=1716566400000,
        body="""Hi Axel,

Invoice #1042 is ready for your review.

  • Invoice number: 1042
  • Amount: $750.00 USD
  • Due date: 2026-06-13 (20 days)
  • Status: PENDING

Acme Billing Team
""",
    ),
    InvoiceSample(
        id="msg-beta-b99",
        from_="Beta Inc Accounting <ar@betainc.com>",
        to="axel@example.com",
        subject="Payment confirmation — Invoice B-99",
        date_ms=1715356800000,
        body="""Hi Axel,

Confirming receipt of your payment of $3,500.00 USD for invoice B-99 on
2026-05-10. Your account balance with Beta Inc is now $0.00.

Thank you!
Beta Inc Accounting
""",
    ),
    InvoiceSample(
        id="msg-gamma-a",
        from_="Gamma Co <invoices@gamma.co>",
        to="axel@example.com",
        subject="Gamma Co Invoice G-2026-A",
        date_ms=1716393600000,
        body="""Axel,

Invoice G-2026-A is now available.

  • Invoice number: G-2026-A
  • Amount: $850.00 USD
  • Due date: 2026-06-03 (10 days)
  • Status: PENDING

Gamma Co
""",
    ),
    InvoiceSample(
        id="msg-gamma-b",
        from_="Gamma Co <invoices@gamma.co>",
        to="axel@example.com",
        subject="Gamma Co Invoice G-2026-B",
        date_ms=1716566400000,
        body="""Axel,

Invoice G-2026-B is now available.

  • Invoice number: G-2026-B
  • Amount: $1,100.00 USD
  • Due date: 2026-06-23 (30 days)
  • Status: PENDING

Gamma Co
""",
    ),
]


DEFAULT_QUESTIONS = [
    "¿Cuánto me debe Acme Corp en total?",
    "¿Qué facturas están vencidas?",
    "¿Quién es el remitente de la factura #1042?",
    "¿Cuál cliente tiene el saldo más grande pendiente?",
    "Resúmeme todas las facturas de Gamma Co.",
    "Did I pay Beta Inc?",
]


# ─── Pipeline ────────────────────────────────────────────────────────


@dataclass
class Answer:
    text: str
    matched_ids: list[str]
    model: str
    total_tokens: int


def to_email(sample: InvoiceSample) -> Email:
    return Email(
        id=sample.id,
        thread_id=f"t-{sample.id}",
        from_=sample.from_,
        to=[sample.to],
        subject=sample.subject,
        body_text=sample.body,
        body_html=f"<pre>{sample.body}</pre>",
        body_clean=normalize(sample.body),
        snippet=sample.body[:80],
        internal_date=sample.date_ms,
        labels=["INBOX"],
    )


def ask(
    question: str,
    cache: EmailCache,
    index: VectorIndex,
    client: LLMClient,
    *,
    k: int = 4,
) -> Answer:
    """Single-turn QA: semantic search top-k → build context → call Llama."""
    hits = index.search(question, k=k)
    relevant = [cache.get(eid) for eid, _ in hits if cache.get(eid) is not None]

    context = "\n\n---\n\n".join(
        f"[Email id={e.id}]\n"
        f"FROM: {e.from_}\n"
        f"SUBJECT: {e.subject}\n"
        f"BODY:\n{e.body_clean}"
        for e in relevant
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Aone, a business assistant. Answer the user's "
                "question using ONLY the emails included below. When "
                "discussing amounts, always cite the invoice number and "
                "sender. If the answer cannot be inferred from the "
                "emails, say so clearly. Reply in the language of the "
                "question."
            ),
        },
        {
            "role": "user",
            "content": f"Emails:\n\n{context}\n\nQuestion: {question}",
        },
    ]

    result = client.complete(messages, max_tokens=500, temperature=0.2)
    return Answer(
        text=result.text,
        matched_ids=[e.id for e in relevant],
        model=result.model,
        total_tokens=result.total_tokens,
    )


def main() -> None:
    print(f"Building cache + index from {len(SAMPLE_INVOICES)} synthetic emails…")
    emails = [to_email(s) for s in SAMPLE_INVOICES]

    cache = EmailCache()
    cache.add_many(emails)

    embedder = LocalEmbedder()
    index = VectorIndex(embedder)
    index.add_many(emails)
    print(
        f"Ready. embedder={embedder.model_name} dims={embedder.dims} "
        f"cache={len(cache)} index={len(index)}\n"
    )

    client = LLMClient()

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = DEFAULT_QUESTIONS

    for q in questions:
        print(f"❓ {q}")
        answer = ask(q, cache, index, client)
        print(f"   matched: {answer.matched_ids}")
        print(f"   model:   {answer.model}  ({answer.total_tokens} tokens)")
        print(f"💬 {answer.text}\n")


if __name__ == "__main__":
    main()
