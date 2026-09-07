"""The MCP surface. Transport only — the behaviour lives in tools.py.

Run it over stdio:

    python -m mcp_local_semantics.server

Or point a host at it. In Claude Desktop's config:

    {
      "mcpServers": {
        "local-semantics": {
          "command": "python",
          "args": ["-m", "mcp_local_semantics.server"]
        }
      }
    }

The encoder loads lazily, so the server starts instantly and only pays the
model load if a tool is actually called.
"""

from __future__ import annotations

import os

# MCP Python SDK v2: FastMCP was renamed to MCPServer and moved.
from mcp.server.mcpserver import MCPServer

from .encoder import HashEncoder, LocalEncoder
from .tools import SemanticTools, ToolError

# MCP_SEMANTICS_ENCODER=hash runs without torch, for smoke-testing a host
# connection before committing to a 90 MB download.
_use_hash = os.environ.get("MCP_SEMANTICS_ENCODER", "").lower() == "hash"
_encoder = HashEncoder() if _use_hash else LocalEncoder()
_tools = SemanticTools(encoder=_encoder)

mcp = MCPServer(
    name="local-semantics",
    title="Local Semantics",
    description=(
        "Embedding tools that run on this machine: search by meaning, compare, "
        "cluster and classify text without an API round trip."
    ),
    version="0.1.0",
)


def _guard(fn, *args, **kwargs):
    """Turn a ToolError into a string the model can act on.

    Raising through the transport ends the exchange. Returning the message
    lets the model correct itself on the next turn, which is the difference
    between a dead conversation and a self-repairing one.
    """
    try:
        return fn(*args, **kwargs)
    except ToolError as exc:
        return {"error": str(exc)}


@mcp.tool(
    name="semantic_search",
    title="Search text by meaning",
    description=(
        "Rank a list of documents by how well each answers a question, using "
        "meaning rather than shared keywords. Use this whenever you have a pile "
        "of text and a question about it: finding the relevant passage in notes, "
        "picking which of several answers fits, or narrowing a long list before "
        "reading it. Runs locally, so it is free to call and safe to call "
        "repeatedly. Set diversity above 0 when the documents contain "
        "near-duplicates and you want coverage rather than the same passage "
        "three times."
    ),
)
def semantic_search(
    query: str, documents: list[str], k: int = 5, diversity: float = 0.0
) -> list[dict] | dict:
    return _guard(_tools.semantic_search, query, documents, k, diversity)


@mcp.tool(
    name="compare_texts",
    title="Compare texts for similarity",
    description=(
        "Measure how similar several pieces of text are to one another and "
        "report the closest and furthest pair. Use this to detect duplicate or "
        "near-duplicate entries, check whether two descriptions mean the same "
        "thing, or find the odd one out in a set."
    ),
)
def compare_texts(texts: list[str]) -> dict:
    return _guard(_tools.compare_texts, texts)


@mcp.tool(
    name="cluster_texts",
    title="Group texts by topic",
    description=(
        "Sort a list of texts into groups by topic without being told what the "
        "topics are. Use this to summarise what themes appear in feedback, "
        "tickets or search results before reading them all, or to split a long "
        "list into coherent batches."
    ),
)
def cluster_texts(texts: list[str], groups: int = 3) -> list[dict] | dict:
    return _guard(_tools.cluster_texts, texts, groups)


@mcp.tool(
    name="classify_texts",
    title="Label texts against categories",
    description=(
        "Assign each text the best-fitting label from a list of categories you "
        "supply in the call. No training and no fixed label set. Use this for "
        "triage, tagging, sentiment, routing, or any time you have categories in "
        "mind and items to sort into them."
    ),
)
def classify_texts(texts: list[str], labels: list[str]) -> list[dict] | dict:
    return _guard(_tools.classify_texts, texts, labels)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
