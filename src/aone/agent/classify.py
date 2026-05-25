"""``classify_intent`` — the first LangGraph node (AONE-401).

Takes the user's natural-language question and returns the most likely
:class:`~aone.agent.intents.Intent`. Uses the model configured by
``AONE_MODEL_CLASSIFICATION`` (default ``groq/llama-3.1-8b-instant``)
via :class:`~aone.llm.client.LLMClient`.

Switching the configured model — to ``groq/llama-3.3-70b-versatile``,
``openai/gpt-4o-mini``, ``anthropic/claude-haiku-4-5``, anything LiteLLM
supports — works without touching this file. The prompt is provider-
agnostic; ADR-005 in action.
"""

from __future__ import annotations

from aone.agent.intents import Intent
from aone.llm.client import LLMClient, Role
from aone.observability.tracing import observe

_SYSTEM_PROMPT = """\
Classify the question into ONE intent. Reply with ONLY the intent name (lowercase, nothing else).

aggregate_amounts: money math (how much, total, sum, balance, owe, debe, suma)
summarize: recap of conversations (summarize, resúmeme, what happened, de qué hablamos)
find_emails: retrieve specific emails (find, show me, busca, muéstrame)
list_contacts: questions about contacts as people (who, quiénes, top clients, inactive)
general_qa: everything else (greetings, capability questions, ambiguous)

Pick the most specific. Reply with one word only."""


@observe(name="classify_intent")
def classify_intent(question: str, client: LLMClient | None = None) -> Intent:
    """Predict the intent of ``question``.

    Args:
        question: the user's natural-language question. Any language
            the underlying model handles is fine.
        client: optional :class:`LLMClient` for testing or for callers
            that want to share an instance. Defaults to a fresh client
            built from the current environment.

    Returns:
        The predicted :class:`Intent`. On unparseable replies, falls
        back to :attr:`Intent.GENERAL_QA`.
    """
    client = client or LLMClient()
    result = client.complete(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        role=Role.CLASSIFICATION,
        max_tokens=15,
        temperature=0.0,
    )
    return Intent.parse(result.text)
