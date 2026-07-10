"""Basic RAG Q&A over the GRC corpus.

query -> dense retrieval (child chunks) -> expand to parent clauses (small-to-big)
-> grounded prompt (answer only from sources, cite, refuse if absent) -> DeepSeek.

The system prompt is English but tells the model to answer in the question's
language, so Chinese questions get Chinese answers with citations.

CLI: uv run python -m rag.qa "Your question on compliance"
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

from ingest.index import COLLECTION, PARENTS_STORE, QDRANT_URL

load_dotenv()  # LLM_API_KEY / LLM_BASE_URL / LLM_MODEL come from .env

MAX_TOKENS = 4096
RETRIEVE_K = 10   # child hits to pull
MAX_PARENTS = 6   # unique parent clauses kept as context

SYSTEM_PROMPT = (
    "You are a GRC compliance assistant. Answer ONLY using the numbered source "
    "excerpts provided below. If the answer is not contained in the sources, say "
    "you do not know — never rely on outside knowledge. Cite the source number "
    "like [n] after every claim. Answer in the same language as the user's question."
)


@dataclass
class ParentCtx:
    id: str
    source: str
    number: str
    title: str
    text: str


def dedup_parents(parent_ids: list[str], store: dict) -> list[ParentCtx]:
    """Unique parents in hit order; source is the id prefix before '#'."""
    seen: set[str] = set()
    out: list[ParentCtx] = []
    for pid in parent_ids:
        if pid in seen or pid not in store:
            continue
        seen.add(pid)
        p = store[pid]
        out.append(
            ParentCtx(
                id=pid,
                source=pid.split("#", 1)[0],
                number=p.get("number", ""),
                title=p.get("title", ""),
                text=p.get("text", ""),
            )
        )
    return out


def build_messages(query: str, parents: list[ParentCtx]) -> list[dict]:
    blocks = []
    for i, p in enumerate(parents, 1):
        label = f"{p.source} {p.number}".strip()
        blocks.append(f"[{i}] ({label})\n{p.text}")
    user = "Sources:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def retrieve(query: str, k: int = RETRIEVE_K) -> list[str]:
    from qdrant_client import QdrantClient

    from ingest.index import embed, get_model

    qv = embed(get_model(), [query])[0]
    client = QdrantClient(url=QDRANT_URL)
    hits = client.query_points(COLLECTION, query=qv.tolist(), limit=k, with_payload=True).points
    return [h.payload["parent_id"] for h in hits]


def generate(messages: list[dict]) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )
    r = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    msg = r.choices[0].message
    return (msg.content or ""), (getattr(msg, "reasoning_content", "") or "")


def answer(query: str) -> dict:
    store = json.loads(PARENTS_STORE.read_text(encoding="utf-8"))
    parents = dedup_parents(retrieve(query), store)[:MAX_PARENTS]
    content, reasoning = generate(build_messages(query, parents))
    return {
        "answer": content,
        "reasoning": reasoning,
        "sources": [
            {"n": i, "source": p.source, "number": p.number, "id": p.id}
            for i, p in enumerate(parents, 1)
        ],
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) < 2:
        print('usage: uv run python -m rag.qa "your question"')
        return

    res = answer(" ".join(sys.argv[1:]))
    print("=== ANSWER ===")
    print(res["answer"])
    print("\n=== SOURCES ===")
    for s in res["sources"]:
        print(f"  [{s['n']}] {s['source']} {s['number']}")


if __name__ == "__main__":
    main()
