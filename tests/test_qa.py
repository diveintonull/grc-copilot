"""Tests for the RAG prompt assembly and parent expansion (P1-07)."""

from rag.qa import ParentCtx, build_messages, dedup_parents


def test_dedup_parents_keeps_unique_parents_in_hit_order():
    store = {
        "dengbao#6.1.1": {"text": "body A", "number": "6.1.1", "title": "6.1.1 x"},
        "dengbao#6.1.2": {"text": "body B", "number": "6.1.2", "title": "6.1.2 y"},
    }
    ids = ["dengbao#6.1.1", "dengbao#6.1.1", "dengbao#6.1.2"]  # duplicate parent from two children
    parents = dedup_parents(ids, store)
    assert [p.id for p in parents] == ["dengbao#6.1.1", "dengbao#6.1.2"]
    assert parents[0].source == "dengbao"  # derived from the id prefix
    assert parents[0].text == "body A"


def test_build_messages_has_english_constraint_and_numbered_sources():
    parents = [
        ParentCtx(id="d#6.1.1", source="GBT+22239-2019", number="6.1.1", title="6.1.1 物理", text="机房门禁"),
    ]
    msgs = build_messages("机房要门禁吗?", parents)
    assert [m["role"] for m in msgs] == ["system", "user"]

    system = msgs[0]["content"].lower()
    assert "only" in system and "cite" in system  # grounding + citation constraint

    user = msgs[1]["content"]
    assert "[1]" in user and "GBT+22239-2019" in user and "6.1.1" in user  # numbered source label
    assert "机房要门禁吗?" in user  # the question is included
