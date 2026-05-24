"""Cross-cutting test setup.

macOS arm64 ships multiple OpenMP runtimes (one from PyTorch, one from
FAISS); when both libraries land in the same process and one initialises
its OMP runtime after the other has cached threadpool state, the
interpreter segfaults during pytest collection.

The fix is to (a) tell Intel OMP to tolerate duplicate runtimes and
(b) ensure torch is the *first* native lib to load. Once torch is up,
faiss can be imported safely.

This file only runs in tests; production code doesn't need the
workaround because production processes don't usually import both
libraries in arbitrary order during a single session.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402, F401  — preload before faiss
