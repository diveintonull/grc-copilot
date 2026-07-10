"""Tests for grounded baseline Q&A over stable SearchHit evidence."""

from __future__ import annotations

from rag import qa as qa_module
from rag.qa import REFUSAL_ANSWER, answer_from_hits, build_messages
from rag.types import RetrievalConfig, SearchHit


def evidence_hit() -> SearchHit:
    return SearchHit(
        chunk_id="GBT-22239@2019#7.1.4.1:0",
        parent_id="GBT-22239@2019#7.1.4.1",
        score=0.91,
        text="应对登录的用户进行身份标识和鉴别。",
        source_id="GBT-22239",
        version="2019",
        section_number="7.1.4.1",
    )


def test_build_messages_contains_only_numbered_search_hit_evidence() -> None:
    messages = build_messages("身份鉴别有什么要求？", [evidence_hit()])

    assert [message["role"] for message in messages] == ["system", "user"]
    system = messages[0]["content"].lower()
    assert "only" in system
    assert "[n]" in system
    user = messages[1]["content"]
    assert "[1]" in user
    assert "GBT-22239@2019#7.1.4.1" in user
    assert evidence_hit().text in user


def test_build_messages_preserves_parent_id_for_unnumbered_sections() -> None:
    hit = SearchHit(
        chunk_id="GDPR@2016-679#h36:0",
        parent_id="GDPR@2016-679#h36",
        score=0.8,
        text="Right to erasure",
        source_id="GDPR",
        version="2016-679",
        section_number="",
    )

    messages = build_messages("What is the right to erasure?", [hit])

    assert "GDPR@2016-679#h36" in messages[1]["content"]


def test_empty_retrieval_refuses_without_calling_generator() -> None:
    def generator(_messages):
        raise AssertionError("generator must not run without evidence")

    result = answer_from_hits("知识库里没有的问题", [], generator=generator)

    assert result["refused"] is True
    assert result["answer"] == REFUSAL_ANSWER
    assert result["sources"] == []


def test_grounded_answer_keeps_numbered_citation_and_source_mapping() -> None:
    seen_messages = []

    def generator(messages):
        seen_messages.extend(messages)
        return "登录用户需要进行身份标识和鉴别。[1]"

    result = answer_from_hits(
        "身份鉴别有什么要求？", [evidence_hit()], generator=generator
    )

    assert seen_messages
    assert result["refused"] is False
    assert result["answer"].endswith("[1]")
    assert result["sources"] == [
        {
            "n": 1,
            "parent_id": "GBT-22239@2019#7.1.4.1",
            "source_id": "GBT-22239",
            "version": "2019",
            "section_number": "7.1.4.1",
            "score": 0.91,
        }
    ]


def test_answer_connects_retrieval_to_the_injected_generator(monkeypatch) -> None:
    config = RetrievalConfig(dense_k=7)
    calls = {}

    def fake_retrieve(query, received_config):
        calls.update(query=query, config=received_config)
        return [evidence_hit()]

    monkeypatch.setattr(qa_module, "retrieve", fake_retrieve)

    result = qa_module.answer(
        "身份鉴别有什么要求？",
        config,
        generator=lambda _messages: "需要身份鉴别。[1]",
    )

    assert calls == {"query": "身份鉴别有什么要求？", "config": config}
    assert result["answer"] == "需要身份鉴别。[1]"
