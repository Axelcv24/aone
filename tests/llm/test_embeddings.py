"""Tests for ``aone.llm.embeddings``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aone.llm.embeddings import (
    Embedder,
    LiteLLMEmbedder,
    LocalEmbedder,
    get_embedder,
)


# ─── Factory ─────────────────────────────────────────────────────────


def test_get_embedder_local_returns_local_embedder() -> None:
    embedder = get_embedder("local", "sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.provider_name == "local"


def test_get_embedder_litellm_returns_litellm_embedder() -> None:
    embedder = get_embedder("litellm", "openai/text-embedding-3-small")
    assert isinstance(embedder, LiteLLMEmbedder)
    assert embedder.provider_name == "litellm"


def test_get_embedder_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder("magic", "some-model")


# ─── Local embedder (real model, session-scoped) ─────────────────────


@pytest.fixture(scope="session")
def local_embedder() -> LocalEmbedder:
    """Loaded once per pytest session — the model load is the slow part."""
    return LocalEmbedder()


def test_local_embedder_default_model_is_minilm() -> None:
    embedder = LocalEmbedder()
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.provider_name == "local"


def test_local_embedder_dims_is_384(local_embedder: LocalEmbedder) -> None:
    assert local_embedder.dims == 384


def test_local_embedder_embeds_batch(local_embedder: LocalEmbedder) -> None:
    vectors = local_embedder.embed(["hello world", "second sample"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


def test_local_embedder_empty_input_returns_empty_list(local_embedder: LocalEmbedder) -> None:
    assert local_embedder.embed([]) == []


def test_local_embedder_similarity_intuition(local_embedder: LocalEmbedder) -> None:
    """Semantically similar texts must be closer than unrelated ones."""
    import math

    a, b, c = local_embedder.embed(
        [
            "I love programming in Python",
            "Coding with Python is my passion",
            "The kitchen smells like fresh bread",
        ]
    )

    def cosine(x: list[float], y: list[float]) -> float:
        dot = sum(xi * yi for xi, yi in zip(x, y, strict=True))
        nx = math.sqrt(sum(xi * xi for xi in x))
        ny = math.sqrt(sum(yi * yi for yi in y))
        return dot / (nx * ny)

    assert cosine(a, b) > cosine(a, c)


# ─── LiteLLM embedder (mocked) ───────────────────────────────────────


@patch("litellm.embedding")
def test_litellm_embedder_embeds_and_caches_dims(mock_embedding: MagicMock) -> None:
    response = MagicMock()
    response.data = [
        {"embedding": [0.1, 0.2, 0.3]},
        {"embedding": [0.4, 0.5, 0.6]},
    ]
    mock_embedding.return_value = response

    embedder = LiteLLMEmbedder("openai/text-embedding-3-small")
    vectors = embedder.embed(["a", "b"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert embedder.dims == 3
    mock_embedding.assert_called_once_with(
        model="openai/text-embedding-3-small", input=["a", "b"]
    )


@patch("litellm.embedding")
def test_litellm_embedder_dims_probes_on_first_access(mock_embedding: MagicMock) -> None:
    response = MagicMock()
    response.data = [{"embedding": [1.0] * 1536}]
    mock_embedding.return_value = response

    embedder = LiteLLMEmbedder("openai/text-embedding-3-small")
    # Access dims before embed() → should trigger a probe call.
    assert embedder.dims == 1536
    mock_embedding.assert_called_once()


@patch("litellm.embedding")
def test_litellm_embedder_empty_input_skips_api(mock_embedding: MagicMock) -> None:
    embedder = LiteLLMEmbedder("openai/text-embedding-3-small")
    assert embedder.embed([]) == []
    mock_embedding.assert_not_called()


# ─── Protocol conformance ────────────────────────────────────────────


def test_both_embedders_satisfy_protocol() -> None:
    """Belt-and-suspenders: assert structural typing matches the Protocol."""
    assert isinstance(LocalEmbedder(), Embedder)
    assert isinstance(LiteLLMEmbedder("any-model"), Embedder)
