"""Stable retrieval types shared by RAG, Agent tools, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    dense_k: int = 20
    sparse_k: int = 20
    fused_k: int = 20
    rerank_k: int = 5
    use_sparse: bool = True
    use_rerank: bool = True
    expand_parent: bool = True


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    parent_id: str
    score: float
    text: str
    source_id: str
    version: str
    section_number: str

