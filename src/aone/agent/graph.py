"""Assembled LangGraph state machine (AONE-410).

Wires the four nodes built in AONE-401, AONE-402, AONE-408, AONE-409
into a linear graph:

    START ─→ classify ─→ select ─→ execute ─→ respond ─→ END

Public surface:

* :class:`AgentState` — the dict shape that flows between nodes.
* :func:`build_agent` — given a populated cache, vector index, and
  LLM client, returns a compiled graph the CLI (AONE-502) and tests
  can invoke with ``agent.invoke({"question": "…"})``.
* :func:`ask` — sugar over invoke for callers that just want the
  :class:`AgentResponse` out without thinking about the state dict.

State assembly is intentionally thin. All the actual logic lives in
the per-node modules so they stay unit-testable in isolation; this
file is the contract between them.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aone.agent.classify import classify_intent
from aone.agent.execute import ExecuteTools, ToolResults
from aone.agent.intents import Intent
from aone.agent.respond import AgentResponse, GenerateResponse
from aone.agent.select_tools import select_tools
from aone.llm.client import LLMClient
from aone.observability.tracing import observe
from aone.storage.cache import EmailCache
from aone.storage.vector import VectorIndex


class AgentState(TypedDict, total=False):
    """The dict that flows between nodes. Fields populated in order:

    * ``question``      — input, set by the caller.
    * ``intent``        — set by ``classify_intent``.
    * ``tool_names``    — set by ``select_tools``.
    * ``tool_results``  — set by ``execute_tools``.
    * ``response``      — set by ``generate_response``; what the user reads.
    """

    question: str
    intent: Intent
    tool_names: list[str]
    tool_results: ToolResults
    response: AgentResponse


# ─── Public builders ─────────────────────────────────────────────────


def build_agent(
    cache: EmailCache,
    index: VectorIndex,
    llm_client: LLMClient,
) -> CompiledStateGraph:
    """Compile and return the four-node Aone agent graph.

    Args:
        cache: populated :class:`EmailCache` (typically loaded from
            ``~/.aone/cache.pkl`` by the CLI).
        index: :class:`VectorIndex` built over ``cache``.
        llm_client: configured :class:`LLMClient`; the graph uses it
            for classification, summarisation, and final response
            generation. Tools that need their own LLM instance
            (``summarize_thread``, ``aggregate_amounts`` validation)
            receive the same client.

    Returns:
        A compiled LangGraph state machine. Invoke with
        ``agent.invoke({"question": "<user question>"})`` to get back
        an :class:`AgentState` with ``response`` populated.
    """
    executor = ExecuteTools(cache, index, llm_client)
    responder = GenerateResponse(llm_client, cache=cache)

    def classify_node(state: AgentState) -> dict:
        intent = classify_intent(state["question"], client=llm_client)
        return {"intent": intent}

    def select_node(state: AgentState) -> dict:
        return {"tool_names": select_tools(state["intent"])}

    def execute_node(state: AgentState) -> dict:
        results = executor(
            tool_names=state["tool_names"],
            question=state["question"],
        )
        return {"tool_results": results}

    def respond_node(state: AgentState) -> dict:
        response = responder(
            question=state["question"],
            intent=state["intent"],
            tool_results=state["tool_results"],
        )
        return {"response": response}

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("select", select_node)
    graph.add_node("execute", execute_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "select")
    graph.add_edge("select", "execute")
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


@observe(name="aone-ask")
def ask(agent: CompiledStateGraph, question: str) -> AgentResponse:
    """Run ``question`` through ``agent`` and return the final response.

    Convenience for callers that don't care about the intermediate
    state — the CLI's ``aone ask`` (AONE-502) uses this.
    """
    final_state: AgentState = agent.invoke({"question": question})  # type: ignore[assignment]
    return final_state["response"]
