# CHATLOG — mcp-local-semantics

> A record of the decisions that produced this repo: what was chosen, what was
> rejected, and what a measurement changed. Written from the session it was
> built in, not reconstructed afterwards.

---

## 7 Sep 2026 — the whole build

Built in one session with Claude Opus 5, directed through a plan → build →
measure → revise loop. Third of three repos built that day, covering the MCP and
model-API side with work that genuinely needed a local model.

### The brief

Cover **MCP** with something where running a model locally is the right answer
rather than a demonstration.

The honest case for local: an agent's *willingness* to use a tool depends on what
it believes the tool costs. A tool that bills per invocation and waits on a
network gets rationed. A free, instant one gets used mid-reasoning. That is a
behavioural difference, not a budget one, and it justified the design.

### Decision: four tools, not one

`semantic_search` alone would have been thinner than the idea deserved.
Compare, cluster and classify are the other three things you want when you have
a pile of text, and all four share one encoder, so the marginal cost of the
extra three was small.

`classify_texts` takes its label set **per call**, which is what makes it useful
to an agent rather than a fixed classifier.

### Decision: an interface, not a mocking library

The whole point of the repo needs a 90 MB model, and a test suite that downloads
one is not a test suite — it depends on a model host being reachable and takes
minutes.

`Encoder` is two members: `dimensions` and `encode`. Deliberately narrower than
sentence-transformers offers, because an interface shaped like the vendor SDK is
a second SDK you maintain.

`HashEncoder` is a signed hashed bag-of-words. **Not a mock** — a real encoder
with real properties: same text gives the same vector, different text a
different one. It has no semantic understanding at all, and that is fine,
because the tests assert properties of the *arithmetic*.

Result: 36 tests in **0.42 s**, no downloads.

### First benchmark — the actual finding

```
semantic_search over 200 documents   1293.5 ms
```

Not the model being slow. Look at the signature:

```python
def semantic_search(query: str, documents: list[str], k: int = 5)
```

**The documents arrive as an argument, on every call.** MCP tools are stateless.
So the obvious implementation embeds all 200, answers, and then embeds all 200
again when the agent asks its next question about the same corpus.

The realistic agent pattern is one corpus and many questions. The naive
implementation is quadratic in exactly the dimension that matters.

### Revision: cache the embeddings

A bounded LRU keyed on the exact string. Embeddings are deterministic for a
fixed model, so **a hit is the same vector, not an approximation** — unusual for
a cache, and it means there is no accuracy tradeoff to argue about.

```
cold cache   1040.4 ms   encodes all 200
warm cache      1.2 ms   encodes only the query
speed-up         893x    at a 95.2% hit rate
```

Two details that are not decoration:

- `dict.fromkeys(texts)` rather than `set(texts)` — de-duplicates while
  preserving order, so a batch containing the same string twice encodes it once
  and the output rows still line up with the input.
- Bounded at 4096 entries. Unbounded is a slow leak in a long-lived stdio server
  an agent keeps feeding new text to.

This is the same pattern as putting a `cache_control` breakpoint on an agent
loop's stable prefix: **in a loop, the expensive part of the call is usually the
part that did not change.** The work is finding the boundary — here the
individual document, because the *set* changes between calls but the members do
not.

### Revision: measure cold and warm in one run

The first benchmark compared a pre-cache run against a post-cache run, which is
not a controlled measurement — different process, different state.

Rewrote it to use a fresh encoder with the model already loaded, so the first
call is genuinely cold and the rest are genuinely warm, in one process, with the
load time excluded.

### Decision: lazy model loading, for an unusual reason

The model takes **21 seconds** to load. The usual argument for lazy loading is
memory; here it is that a stdio MCP server is launched by the host, often at
startup, and may never have a tool called on it.

Eager loading charges every host launch 21 seconds for a capability the user
might not use.

### Decision: errors come back as results

```python
raise ToolError(
    f"texts has {len(texts)} items; the limit is {MAX_TEXTS}. "
    f"Split the batch and call again."
)
```

Caught at the MCP boundary and returned. Raising through the transport ends the
exchange; returning lets the model fix it next turn. Every message says what to
do, not only what was wrong, and there are tests asserting the instruction is
present in the string.

### Things that went wrong

- **The MCP Python SDK is on v2.** `from mcp.server.fastmcp import FastMCP`
  fails; it is now `from mcp.server.mcpserver import MCPServer`. The error
  message is unusually good — it names the rename and links the migration guide.
  Chose to migrate rather than pin `mcp<2`.
- **Nested shell escaping mangled `bench.py`** on one edit — `\n` inside a
  heredoc inside a Python string became a literal newline. Rewrote the file
  whole instead of patching it.

### Deliberately not done

- **No persistence or vector store.** If the corpus does not fit in a tool
  argument, you want a database and this is the wrong tool.
- **No HTTP transport.** Adding a network surface reintroduces exactly the
  latency the repo exists to remove.
- **No reranking.** It lives in the sibling repo `rag-reranker-lab`, where it
  can be measured rather than assumed — and where the measurement showed it
  barely helping.

---

## Conventions for future entries

Each entry should record **what changed, what it was measured against, and what
was rejected.** An entry that only says what was added is a commit message and
belongs in git instead.
