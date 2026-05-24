"""Provider-agnostic embedding interface (ADR-005).

Two implementations:

* :class:`LocalEmbedder` — runs ``sentence-transformers`` on CPU. Free,
  no network call after the model is cached on disk. Default for v0.
* :class:`LiteLLMEmbedder` — routes through LiteLLM, so anything LiteLLM
  supports (OpenAI ``text-embedding-3-small``, Voyage, Cohere, Gemini,
  etc.) is reachable by setting ``AONE_EMBEDDING_PROVIDER=litellm`` and
  the matching ``AONE_EMBEDDING_MODEL``.

Both embedders are lazy: importing the module is cheap, the model is
loaded on first use.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Structural type for anything that produces embeddings."""

    provider_name: str
    model_name: str

    @property
    def dims(self) -> int:
        """Embedding dimensionality. May trigger lazy model load."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class LocalEmbedder:
    """Embeddings via ``sentence-transformers`` running on CPU."""

    provider_name = "local"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: object | None = None
        self._dims: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            self._model = model
            # `get_embedding_dimension` is the post-5.x name; fall back to
            # the old name for older installs.
            dim_method = getattr(
                model, "get_embedding_dimension", None
            ) or model.get_sentence_embedding_dimension
            self._dims = int(dim_method())

    @property
    def dims(self) -> int:
        self._ensure_loaded()
        assert self._dims is not None
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._model is not None
        # show_progress_bar=False so test runs don't spam stdout.
        vectors = self._model.encode(texts, show_progress_bar=False)  # type: ignore[attr-defined]
        return [vec.tolist() for vec in vectors]


class LiteLLMEmbedder:
    """Embeddings via LiteLLM (OpenAI / Voyage / Cohere / Gemini / …)."""

    provider_name = "litellm"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._dims: int | None = None

    @property
    def dims(self) -> int:
        if self._dims is None:
            # Probe with a single short string; the result tells us the
            # vector size. One-time cost per process.
            self.embed(["probe"])
        assert self._dims is not None
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import litellm

        response = litellm.embedding(model=self.model_name, input=texts)
        vectors = [item["embedding"] for item in response.data]
        if self._dims is None and vectors:
            self._dims = len(vectors[0])
        return vectors


def get_embedder(provider: str, model: str) -> Embedder:
    """Return an embedder for ``(provider, model)``.

    Args:
        provider: ``"local"`` or ``"litellm"`` (the value of
            ``AONE_EMBEDDING_PROVIDER``).
        model: model identifier passed straight to the implementation.

    Raises:
        ValueError: if ``provider`` is unknown.
    """
    if provider == "local":
        return LocalEmbedder(model_name=model)
    if provider == "litellm":
        return LiteLLMEmbedder(model_name=model)
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        f"Expected 'local' or 'litellm'."
    )
