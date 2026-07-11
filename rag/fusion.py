"""Deterministic Reciprocal Rank Fusion for Dense and Sparse candidates."""

from __future__ import annotations

from dataclasses import replace

from rag.types import SearchHit


def reciprocal_rank_fusion(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    *,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse rankings by chunk id and retain each source's one-based rank."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    hits: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    seen_counter = 0

    for ranking, ranks in (
        (dense_hits, dense_ranks),
        (sparse_hits, sparse_ranks),
    ):
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk_id not in hits:
                hits[hit.chunk_id] = hit
                first_seen[hit.chunk_id] = seen_counter
                seen_counter += 1
            if hit.chunk_id in ranks:
                continue
            ranks[hit.chunk_id] = rank
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )

    ordered_ids = sorted(
        hits,
        key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]),
    )
    if limit is not None:
        ordered_ids = ordered_ids[:limit]
    return [
        replace(
            hits[chunk_id],
            score=scores[chunk_id],
            dense_rank=dense_ranks.get(chunk_id),
            sparse_rank=sparse_ranks.get(chunk_id),
        )
        for chunk_id in ordered_ids
    ]
