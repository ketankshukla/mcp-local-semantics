# SPEC — mcp-local-semantics

> A specification, not a tutorial. It says what this repo is for, what it
> deliberately does not do, and what must stay true for a change to be correct.

**Status:** complete for its purpose · **Version:** 0.1.0 · **Last verified:** 7 Sep 2026

---

## 1. Purpose

Give an agent semantic operations it can use **freely, mid-reasoning**, by running
them on the machine the agent is already on.

> An agent will call a free, instant tool inside a loop. It will not call one
> that bills per invocation and waits on a network.

The money saved is rounding error — embedding 18,000 tokens at list price is
about $0.0004. The behavioural change is the point: local moves semantic search
from a resource to be rationed into an operation the agent reaches for.

## 2. Scope

### In scope
- An MCP server over stdio, MCP Python SDK v2
- Four tools: `semantic_search`, `compare_texts`, `cluster_texts`, `classify_texts`
- A local sentence encoder with a bounded embedding cache
- Pure-NumPy implementations of top-k, MMR, k-means and zero-shot classification
- A dependency-free encoder for tests and CI

### Explicitly out of scope
- **Persistence.** No index is stored between calls. Documents arrive as a tool
  argument, per the protocol. The cache is an optimisation, not storage.
- **A vector database.** If your corpus does not fit in a tool argument, you
  want a database and this is the wrong tool.
- **Generation.** No LLM is called. This server hands back vectors and rankings.
- **Model training or fine-tuning.** The encoder is used as published.
- **HTTP or SSE transport.** stdio only. Adding a network surface reintroduces
  the latency this repo exists to remove.
- **Reranking.** Deliberately in the sibling repo `rag-reranker-lab`, where it
  can be measured rather than assumed.

## 3. Architecture

```
                        MCP host (Claude Desktop, an agent, …)
                                     │  stdio
                                     ▼
server.py        transport only — tool registration, error containment
                                     │
                                     ▼
tools.py         validation, guard rails, result shaping
                                     │
        ┌────────────────────────────┴───────────────────────┐
        ▼                                                    ▼
encoder.py                                            semantics.py
  Encoder protocol                                      pure NumPy
  LocalEncoder  (PyTorch + LRU cache)                   top_k · mmr
  HashEncoder   (deterministic, no torch)               cluster · classify
```

**The layering rule:** `semantics.py` knows nothing about models or MCP.
`tools.py` knows nothing about MCP. `server.py` contains no logic. Each layer is
testable without the one above it.

## 4. Interfaces

### Encoder protocol — the seam

```python
class Encoder(Protocol):
    dimensions: int
    def encode(self, texts: list[str]) -> np.ndarray: ...
```

Two methods' worth of surface, deliberately narrower than sentence-transformers
offers. An interface shaped like the vendor SDK is a second SDK you maintain;
one shaped like your own call site cannot drift.

| implementation | backing | used for |
|---|---|---|
| `LocalEncoder` | `all-MiniLM-L6-v2` via PyTorch, 384-d | production |
| `HashEncoder` | signed hashed bag-of-words, 256-d | tests, CI, smoke tests |

### The four tools

| tool | signature | returns |
|---|---|---|
| `semantic_search` | `(query, documents, k=5, diversity=0.0)` | ranked matches with index, score, text |
| `compare_texts` | `(texts)` | full similarity matrix + closest and furthest pair |
| `cluster_texts` | `(texts, groups=3)` | groups with member indices and texts |
| `classify_texts` | `(texts, labels)` | one label + score per text |

### Configuration

| variable | effect |
|---|---|
| `MCP_SEMANTICS_ENCODER=hash` | run without torch; tools respond, semantics are meaningless |

### Limits

`MAX_TEXTS = 512` · `MAX_CHARS = 8_000` per text.

## 5. Invariants

1. **The encoder is injected, never imported at module scope in `semantics.py`
   or `tools.py`.** Breaking this makes the suite depend on a model download.
2. **Model loading is lazy.** A stdio server's startup cost is paid by a user
   who has not asked for anything yet; 21 s of eager load is unacceptable.
3. **The embedding cache is bounded and LRU.** Unbounded is a slow leak in a
   long-lived server an agent keeps feeding new text to.
4. **A cache hit returns the identical vector**, not an approximation.
   Embeddings are deterministic for a fixed model, so there is no accuracy
   tradeoff to reason about.
5. **No similarity threshold in `top_k`.** Best *k* by rank, always. Callers
   wanting a floor apply it to scores they have seen.
6. **Clustering is deterministic for a given seed.** A tool an agent calls twice
   on the same input and gets two answers from is a tool it cannot reason about.
7. **`ToolError` is returned as a result, never raised through the transport.**
   Raising ends the exchange; returning lets the model correct itself next turn.
8. **Every guard-rail message says what to do**, not only what was wrong —
   "Split the batch and call again", "Chunk it before embedding". There are
   tests asserting the instruction is present.
9. **Tool descriptions name the *situations* the tool applies to**, not what the
   function does. A vague description means the tool is never selected.
10. **The test suite performs no network I/O and downloads no model.**

## 6. Verification

| | |
|---|---|
| tests | 36, `pytest -q`, **0.42 s**, no network |
| coverage | vector maths, all four tools, guard rails, `HashEncoder` |
| not covered | `LocalEncoder` output quality — exercised by `scripts/bench.py` |
| CI | GitHub Actions, `numpy pytest` only |
| benchmark | `python scripts/bench.py` |

Tests assert properties: that a zero row does not become NaN, that `top_k`
applies no floor, that MMR with `diversity` set reaches past near-duplicates,
that clustering is reproducible under a seed.

## 7. Results this repo produced

1,000 documents · ~18,000 tokens · CPU · `all-MiniLM-L6-v2`

| | |
|---|---|
| model load (once, lazy) | 20.99 s |
| encode 1,000 documents | 5.58 s — 179 docs/s, 3,265 tok/s |
| `semantic_search` over 200, cold cache | **1,040.4 ms** |
| `semantic_search` over 200, warm cache | **1.2 ms** |
| speed-up | **893×** at a 95.2% hit rate |
| same call via `HashEncoder` | 16.4 ms |

**Finding:** an MCP tool is stateless, so documents arrive as an argument on
every call and the naive implementation re-embeds the whole corpus each time.
The realistic pattern is one corpus and many questions, which makes that the
dominant cost.

## 8. Known limitations

- **CPU throughput.** ~179 docs/s. Corpora in the tens of thousands want a GPU
  or a precomputed index, which means a different design.
- **The cache is keyed on the exact string.** A single character difference is a
  miss. Correct, but it means callers who reformat their documents between calls
  get no benefit.
- **Cache is per-process.** A host that restarts the server loses it.
- **`all-MiniLM-L6-v2` is English and general-domain.** Technical jargon,
  identifiers and non-English text will degrade.
- **k-means with cosine on unit vectors is spherical k-means**, which assumes
  roughly equal-sized, roughly spherical clusters. Uneven topic distributions
  cluster poorly.
- **No streaming.** A 512-document call returns in one response.

## 9. Extending it

Swap the model: change `LocalEncoder.model_name`. Anything
sentence-transformers loads will work; `dimensions` is read from the model.

Add a tool: add a method to `SemanticTools`, register it in `server.py` with a
description that names the situations it applies to, and add a test using
`HashEncoder`.

Persist the cache: implement `Encoder` over a store. The protocol is two
members, and nothing above it changes.

## 10. Related

- Post: [A stateless MCP tool re-embeds its corpus on every call](https://ketanshukla.dev/blog/a-stateless-mcp-tool-re-embeds-everything)
- Related: [An MCP server is one Next.js route handler](https://ketanshukla.dev/blog/an-mcp-server-is-one-route-handler)
- Related: [One cache breakpoint cut my agent loop's input cost by 47%](https://ketanshukla.dev/blog/prompt-caching-cut-input-cost-47-percent)
- Decision history: [CHATLOG.md](CHATLOG.md)
