"""Measure what the local encoder actually does on this machine.

Measured here:
  - model load time (paid once, lazily)
  - encoding throughput on CPU
  - semantic_search latency, cold cache and warm

Computed, not measured, and labelled as such wherever it appears:
  - what the same token volume would cost against a hosted embedding API,
    from published list price

    python scripts/bench.py
    python scripts/bench.py --n 2000 --repeats 40
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_local_semantics.encoder import HashEncoder, LocalEncoder  # noqa: E402
from mcp_local_semantics.tools import SemanticTools  # noqa: E402

# OpenAI text-embedding-3-small list price, USD per million tokens, Sep 2026.
# Change it if it has moved; nothing else in this file depends on the value.
API_USD_PER_MTOK = 0.02

SAMPLE = [
    "the deployment runbook covers rollback steps for each service",
    "invoice line items are prorated when a plan changes mid-period",
    "error rates spiked after the release went out at 02:40",
    "parental leave policy for employees in their first year",
    "the retry envelope carries an idempotency key on every attempt",
    "customer asked for a refund on the unused portion of the term",
    "connection pool exhaustion under load caused the timeouts",
    "find a slot that works across London and San Diego next week",
]

QUERIES = [
    "why did the deploy break checkout",
    "how is a mid-cycle plan change billed",
    "what causes the timeouts under load",
    "how much leave do new joiners get",
]


def corpus(n: int) -> list[str]:
    return [f"{SAMPLE[i % len(SAMPLE)]} (variant {i})" for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="documents to encode")
    ap.add_argument("--repeats", type=int, default=20, help="search calls to time")
    args = ap.parse_args()

    docs = corpus(args.n)
    # ~4 characters per token is the usual rule of thumb for English.
    tokens = sum(len(d) for d in docs) // 4
    print(f"{args.n} documents · ~{tokens:,} tokens · CPU\n")

    # ---------------------------------------------------------------- load
    encoder = LocalEncoder()
    started = time.perf_counter()
    encoder.encode(["warm up"])
    print(f"  model load (once, lazy)      {time.perf_counter() - started:8.2f} s")

    started = time.perf_counter()
    vectors = encoder.encode(docs)
    encode = time.perf_counter() - started
    print(
        f"  encode {args.n:>5} documents      {encode:8.2f} s"
        f"   ({args.n / encode:,.0f} docs/s, {tokens / encode:,.0f} tok/s)"
    )
    print(f"  vector dimensions            {vectors.shape[1]:8d}")

    # --------------------------------------------------- cold vs warm cache
    # A fresh encoder that has already loaded the model but cached nothing, so
    # the first call pays for encoding the corpus and later calls do not. This
    # is the realistic agent pattern: one corpus, many questions.
    fresh = LocalEncoder()
    fresh._load()  # noqa: SLF001 - exclude load time from the measurement
    tools = SemanticTools(encoder=fresh)
    small = docs[:200]

    started = time.perf_counter()
    tools.semantic_search(QUERIES[0], small, k=5)
    cold = (time.perf_counter() - started) * 1000

    warm_times = []
    for i in range(args.repeats):
        q = QUERIES[i % len(QUERIES)]
        started = time.perf_counter()
        tools.semantic_search(q, small, k=5)
        warm_times.append((time.perf_counter() - started) * 1000)
    warm = statistics.median(warm_times)

    print("\n  semantic_search over 200 documents")
    print(f"    first call,  cold cache  {cold:8.1f} ms   encodes all 200")
    print(f"    later calls, warm cache  {warm:8.1f} ms   encodes only the query")
    print(f"    speed-up                 {cold / warm:8.0f}x")
    print(f"    cache hit rate           {fresh.hit_rate:8.1%}")

    hashed = SemanticTools(encoder=HashEncoder())
    started = time.perf_counter()
    for _ in range(args.repeats):
        hashed.semantic_search(QUERIES[0], small, k=5)
    hash_ms = (time.perf_counter() - started) / args.repeats * 1000
    print(f"\n  same call, HashEncoder     {hash_ms:8.1f} ms   (no model, the CI path)")

    api_cost = tokens / 1_000_000 * API_USD_PER_MTOK
    print(
        f"\n  Computed, not measured: embedding {tokens:,} tokens against a hosted"
        f"\n  API at ${API_USD_PER_MTOK}/Mtok would cost ${api_cost:.4f}."
        f" Locally it costs the {encode:.1f}s above and nothing else."
    )
    print(
        "\n  The dollar figure is not the point. The point is that an agent will"
        "\n  call a free, fast tool inside a loop, and will not call one that"
        "\n  bills per invocation and waits on a network."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
