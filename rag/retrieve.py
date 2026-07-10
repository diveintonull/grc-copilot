"""Unified retrieval entry point with small-to-big parent expansion."""

from __future__ import annotations

import json

from ingest.index import PARENTS_STORE
from rag.dense import search_dense
from rag.types import RetrievalConfig, SearchHit


def expand_parent_hits(
    child_hits: list[SearchHit], parent_store: dict
) -> list[SearchHit]:
    """Replace child text with full parent text and keep the first hit per parent."""
    seen: set[str] = set()
    expanded: list[SearchHit] = []
    for hit in child_hits:
        if hit.parent_id in seen or hit.parent_id not in parent_store:
            continue
        seen.add(hit.parent_id)
        expanded.append(
            SearchHit(
                chunk_id=hit.chunk_id,
                parent_id=hit.parent_id,
                score=hit.score,
                text=parent_store[hit.parent_id]["text"],
                source_id=hit.source_id,
                version=hit.version,
                section_number=hit.section_number,
            )
        )
    return expanded


def retrieve(query: str, config: RetrievalConfig) -> list[SearchHit]:
    """Run the Dense baseline and optionally expand matching children to parents."""
    child_hits = search_dense(query, k=config.dense_k)
    if not config.expand_parent:
        return child_hits[: config.fused_k]
    parent_store = json.loads(PARENTS_STORE.read_text(encoding="utf-8"))
    return expand_parent_hits(child_hits, parent_store)[: config.fused_k]
