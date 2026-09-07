"""The seam between the maths and the model.

`Encoder` is the whole interface: text in, an array out. That is deliberately
narrower than anything sentence-transformers offers — an interface shaped like
the vendor's SDK is a second SDK you now maintain, whereas one shaped like your
own call site is four lines and cannot drift.

Two implementations:

  LocalEncoder  the real one, a MiniLM bi-encoder on CPU via PyTorch
  HashEncoder   deterministic, dependency-free, for tests and CI

HashEncoder is not a mock in the usual sense. It is a real encoder with real
properties — same text gives the same vector, different text gives a different
one — which is enough to exercise every code path in semantics.py without
downloading weights.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    dimensions: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class LocalEncoder:
    """A sentence-transformers bi-encoder, loaded once and kept warm.

    Loading is lazy so importing this module costs nothing: an MCP server
    starts, advertises its tools, and only pays the model load if a tool is
    actually called. On a stdio server that a host may launch and never use,
    that is the difference between instant and four seconds.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    cache_size: int = 4096

    _model: object = field(default=None, repr=False)
    _cache: OrderedDict[str, np.ndarray] = field(
        default_factory=OrderedDict, repr=False
    )
    hits: int = 0
    misses: int = 0

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode, reusing vectors already computed for identical text.

        An MCP tool is stateless: the documents arrive as an argument on every
        call, so a naive implementation re-embeds the whole corpus each time an
        agent searches it. That is the dominant cost in practice, because the
        realistic pattern is one corpus and many questions.

        Keyed on the exact string. Embeddings are deterministic for a given
        model, so a hit is not an approximation — it is the same vector.
        """
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        self.hits += len(texts) - sum(1 for t in texts if t in missing)
        self.misses += len(missing)

        if missing:
            model = self._load()
            fresh = np.asarray(
                model.encode(
                    missing, normalize_embeddings=True, show_progress_bar=False
                ),
                dtype=np.float32,
            )
            for text, vector in zip(missing, fresh, strict=True):
                self._cache[text] = vector
            # Bounded, least-recently-used. Unbounded would be a slow leak in a
            # long-lived server that an agent keeps feeding new text to.
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        out = np.empty((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            vector = self._cache[text]
            self._cache.move_to_end(text)
            out[i] = vector
        return out

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def loaded(self) -> bool:
        return self._model is not None


@dataclass
class HashEncoder:
    """A hashed bag-of-words encoder. No downloads, no torch, deterministic.

    It has no semantic understanding at all — "car" and "automobile" are
    orthogonal to it. That is fine for what it is for: proving that top_k
    orders correctly, that MMR de-duplicates, that clustering converges. Those
    are properties of the arithmetic, not of the embedding.

    It also gives the tools a usable fallback when torch is not installed,
    rather than an import error at call time.
    """

    dimensions: int = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                # Signed hashing: the second half of the digest picks a sign, so
                # unrelated tokens landing in the same bucket cancel on average
                # instead of always reinforcing.
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, bucket] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
