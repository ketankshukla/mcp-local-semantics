# 🧠 MCP Local Semantics

**Four embedding tools an agent can call in a loop, because they cost nothing.**

An MCP server that runs a sentence encoder on the machine the agent is already on. Search by meaning, compare texts, cluster them, classify against labels you supply per call — no embedding API, no key, no network, no per-invocation bill.

The interesting part is not that a local model works. It is that **a stateless tool re-embeds its whole corpus on every call**, which made the first version 1,040 ms per search. Caching the embeddings took it to 1.2 ms.

🔧 **The MCP surface:** [`src/mcp_local_semantics/server.py`](src/mcp_local_semantics/server.py)
🧮 **The maths, pure NumPy:** [`src/mcp_local_semantics/semantics.py`](src/mcp_local_semantics/semantics.py)
📝 **Related write-up:** [An MCP server is one Next.js route handler](https://ketanshukla.dev/blog/an-mcp-server-is-one-route-handler)

---

## Measured on CPU

1,000 documents · ~18,000 tokens · `all-MiniLM-L6-v2` · no GPU

| | |
|---|---|
| model load (once, lazily) | 20.99 s |
| encode 1,000 documents | 5.58 s — **179 docs/s, 3,265 tok/s** |
| `semantic_search` over 200, **cold cache** | **1040.4 ms** |
| `semantic_search` over 200, **warm cache** | **1.2 ms** |
| speed-up | **893×** (95.2% hit rate) |
| same call via `HashEncoder` (the CI path) | 16.4 ms |

Reproduce with `python scripts/bench.py`.

### The 893× is the whole design problem

An MCP tool is stateless. The documents arrive as an *argument*, on every call. So the obvious implementation embeds all 200 of them every time the agent asks a question — and the realistic agent pattern is one corpus and many questions.

```python
# LocalEncoder.encode — keyed on the exact string.
missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
```

Embeddings are deterministic for a given model, so a cache hit is not an approximation — it is the same vector. The cache is bounded and least-recently-used, because an unbounded one is a slow leak in a long-lived server that an agent keeps feeding new text to.

This is the same shape as [putting a cache breakpoint on an agent loop's stable prefix](https://ketanshukla.dev/blog/prompt-caching-cut-input-cost-47-percent): the expensive part of the call is the part that did not change.

### On the cost comparison

Embedding those 18,000 tokens against a hosted API at list price would cost about **$0.0004**. That is nothing, and quoting it as the headline would be dishonest.

The number that matters is the other one. **An agent will call a free, instant tool inside a loop. It will not call one that bills per invocation and waits on a network** — and neither will you, when you are deciding whether to let it. Local moves semantic search from a resource to be rationed into an operation the agent can use freely.

The API figure above is computed from published list price and labelled as computed everywhere it appears. It was not measured.

---

## The tools

| tool | what it is for |
|---|---|
| `semantic_search` | rank documents by how well each answers a question, by meaning rather than shared words |
| `compare_texts` | pairwise similarity, plus the closest and furthest pair — duplicate detection, odd-one-out |
| `cluster_texts` | group texts by topic without being told the topics |
| `classify_texts` | assign the best label from a list you supply in the call — no training, no fixed label set |

### The descriptions are the interesting part

A vague tool description means the model never selects the tool at all. Each description here names the *situations* it applies to, not what the function does:

> Rank a list of documents by how well each answers a question, using meaning rather than shared keywords. **Use this whenever you have a pile of text and a question about it**: finding the relevant passage in notes, picking which of several answers fits, or narrowing a long list before reading it. **Runs locally, so it is free to call and safe to call repeatedly.**

That last sentence is doing real work. It tells the model something true about the tool's cost, which changes how willing it is to reach for it mid-reasoning.

---

## Running it

```bash
pip install -r requirements.txt
python -m mcp_local_semantics.server        # stdio
```

Claude Desktop:

```json
{
  "mcpServers": {
    "local-semantics": {
      "command": "python",
      "args": ["-m", "mcp_local_semantics.server"]
    }
  }
}
```

Set `MCP_SEMANTICS_ENCODER=hash` to run without torch — useful for smoke-testing the host connection before committing to a 90 MB download.

```bash
pytest -q                  # 36 tests, under a second, no downloads
python scripts/bench.py    # the table above
```

> Built against **MCP Python SDK v2**, where `FastMCP` was renamed to `MCPServer` and moved to `mcp.server.mcpserver`. Code written for v1 needs either that change or a `mcp<2` pin.

---

## Design notes

### The encoder is an interface, so the maths is testable

`Encoder` is four lines: text in, an array out. Deliberately narrower than anything sentence-transformers offers — an interface shaped like the vendor SDK is a second SDK you now maintain.

Two implementations. `LocalEncoder` is the real one. `HashEncoder` is a signed hashed bag-of-words: deterministic, no torch, no network. It has no semantic understanding at all — "car" and "automobile" are orthogonal to it — and that is fine, because the test suite is asserting properties of the *arithmetic*: that top-k orders correctly, that MMR de-duplicates, that clustering converges and is deterministic.

The whole suite runs in **0.42 s** with no downloads, so CI does not depend on a model host being reachable. Same separation that made [an iteration cap testable against a mocked model client](https://ketanshukla.dev/blog/unit-test-an-iteration-cap-without-a-model).

### No similarity threshold, anywhere

`top_k` returns the best *k* regardless of score. A fixed cutoff is not scale-invariant across queries — too strict for short questions, too loose for long ones — and when it is too strict it **fails silently**, which is [the bug this came out of](https://ketanshukla.dev/blog/deleting-the-similarity-threshold). There is a test asserting no floor is applied. Callers wanting one can apply it to the scores they get back, having seen them.

### Clustering is deterministic on purpose

K-means with k-means++ seeding, written out rather than imported so the tool has no scikit-learn dependency at runtime. Seeded, because **a tool an agent calls twice on the same input and gets two answers from is a tool the agent cannot reason about.** There is a test for it.

k-means++ rather than uniform seeding because uniform routinely puts two centres inside one cluster and leaves another unclaimed.

### Errors come back as results, not exceptions

```python
raise ToolError(
    f"texts has {len(texts)} items; the limit is {MAX_TEXTS}. "
    f"Split the batch and call again."
)
```

Caught at the MCP boundary and returned as a result. Raising through the transport ends the exchange; returning the message lets the model fix it on the next turn. Every guard-rail message says what to do, not just what was wrong — and there are tests asserting the messages contain the instruction.

### MMR for when the corpus has near-duplicates

Plain top-k over three paraphrases of the same passage returns all three and tells the model one thing. `diversity` above 0 penalises a candidate by its similarity to what is already selected. At 0 it is exactly top-k; at 1 it ignores the query entirely.

---

## Stack

Python · PyTorch (CPU) · sentence-transformers · MCP Python SDK v2 · NumPy · pytest

MIT licensed. Built by [Ketan Shukla](https://ketanshukla.dev).
