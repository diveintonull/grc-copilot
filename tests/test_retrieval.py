"""Contract tests for Dense retrieval and parent expansion."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rag import dense
from rag import retrieve as retrieve_module
from rag.retrieve import expand_parent_hits
from rag.types import RetrievalConfig, SearchHit


def child_hit(
    chunk_id: str,
    parent_id: str,
    *,
    score: float,
    text: str = "child text",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        parent_id=parent_id,
        score=score,
        text=text,
        source_id="GBT-22239",
        version="2019",
        section_number="7.1.4.1",
    )


def test_retrieval_config_has_stable_dense_baseline_defaults() -> None:
    config = RetrievalConfig()

    assert config.dense_k == 20
    assert config.fused_k == 20
    assert config.expand_parent is True


def test_expand_parent_hits_deduplicates_in_first_hit_order() -> None:
    hits = [
        child_hit("p1:0", "p1", score=0.9),
        child_hit("p1:1", "p1", score=0.8),
        child_hit("p2:0", "p2", score=0.7),
    ]
    store = {
        "p1": {"text": "full parent one"},
        "p2": {"text": "full parent two"},
    }

    expanded = expand_parent_hits(hits, store)

    assert [hit.parent_id for hit in expanded] == ["p1", "p2"]
    assert [hit.chunk_id for hit in expanded] == ["p1:0", "p2:0"]
    assert [hit.text for hit in expanded] == ["full parent one", "full parent two"]


def test_dense_search_returns_search_hits_from_qdrant_payload(monkeypatch) -> None:
    point = SimpleNamespace(
        score=0.87,
        payload={
            "chunk_id": "GBT-22239@2019#7.1.4.1:0",
            "parent_id": "GBT-22239@2019#7.1.4.1",
            "text": "身份鉴别子块",
            "source_id": "GBT-22239",
            "version": "2019",
            "section_number": "7.1.4.1",
        },
    )
    calls = {}

    class Vector:
        def tolist(self):
            return [0.1, 0.2]

    class Client:
        def query_points(self, collection, **kwargs):
            calls.update(collection=collection, **kwargs)
            return SimpleNamespace(points=[point])

    monkeypatch.setattr(dense, "get_model", lambda: object())
    monkeypatch.setattr(dense, "embed", lambda _model, _texts: [Vector()])
    monkeypatch.setattr(dense, "_client", lambda: Client())
    if hasattr(dense, "_get_cached_model"):
        dense._get_cached_model.cache_clear()

    hits = dense.search_dense("身份鉴别", k=7)

    assert hits == [
        SearchHit(
            chunk_id="GBT-22239@2019#7.1.4.1:0",
            parent_id="GBT-22239@2019#7.1.4.1",
            score=0.87,
            text="身份鉴别子块",
            source_id="GBT-22239",
            version="2019",
            section_number="7.1.4.1",
        )
    ]
    assert calls["limit"] == 7
    assert calls["with_payload"] is True


def test_dense_search_reuses_the_embedding_model(monkeypatch) -> None:
    loads = 0

    class Vector:
        def tolist(self):
            return [0.1]

    class Client:
        def query_points(self, _collection, **_kwargs):
            return SimpleNamespace(points=[])

    def load_model():
        nonlocal loads
        loads += 1
        return object()

    monkeypatch.setattr(dense, "get_model", load_model)
    monkeypatch.setattr(dense, "embed", lambda _model, _texts: [Vector()])
    monkeypatch.setattr(dense, "_client", lambda: Client())
    dense._get_cached_model.cache_clear()

    dense.search_dense("first", k=1)
    dense.search_dense("second", k=1)

    assert loads == 1


def test_retrieve_uses_config_and_expands_parents(monkeypatch, tmp_path) -> None:
    parent_store = tmp_path / "parents.json"
    parent_store.write_text(
        json.dumps({"p1": {"text": "full parent"}}), encoding="utf-8"
    )
    calls = {}

    def fake_search(query: str, *, k: int) -> list[SearchHit]:
        calls.update(query=query, k=k)
        return [child_hit("p1:0", "p1", score=0.9)]

    monkeypatch.setattr(retrieve_module, "PARENTS_STORE", parent_store)
    monkeypatch.setattr(retrieve_module, "search_dense", fake_search)

    hits = retrieve_module.retrieve(
        "身份鉴别", RetrievalConfig(dense_k=7, fused_k=5)
    )

    assert calls == {"query": "身份鉴别", "k": 7}
    assert [hit.text for hit in hits] == ["full parent"]
