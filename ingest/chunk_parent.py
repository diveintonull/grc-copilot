"""Parent-child segmenter.

Our parsed Markdown has flat `##` headings — the real hierarchy lives in the
*numbering* (`6.1.9.1`) for GB/T standards and in `第X条` markers for PRC statutes,
not in the `#` depth. So reconstruct sections from those, not from heading levels.

A parent (Section) is a coherent unit — a numbered subsection (等保) or an article
(法律) — stored whole and NOT embedded. A child is a fixed-size chunk of that
section's text, tagged with `parent_id`; only children go into the vector store,
and a child hit is expanded to its parent for full context (small-to-big).

CLI: uv run python -m ingest.chunk_parent   (writes 等保 parent-child JSON)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ingest.chunk import Chunk, chunk_document

PARSED = Path("data/parsed")
DENGBAO = PARSED / "GBT+22239-2019.md"

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$")
_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)")
_ARTICLE_RE = re.compile(r"^\*{0,2}(第[一二三四五六七八九十百千零〇两]+条)\*{0,2}")


@dataclass
class Section:
    id: str
    title: str
    number: str
    level: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def _heading_title(line: str) -> str | None:
    match = _HEADING_RE.match(line)
    return match.group(1) if match else None


def segment_by_headings(text: str, *, doc_id: str) -> list[Section]:
    """Split Markdown into sections at `#` headings.

    Each heading starts a section whose text runs to the next heading (last one
    to EOF). `number` = the leading dotted number in the heading title; `level` =
    its number of dotted segments (0 if unnumbered). Content before the first
    heading is dropped. id = f"{doc_id}#{number}" (or f"{doc_id}#h{idx}").
    """
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if _heading_title(line) is not None]
    ranges = [
        (start, starts[index + 1] if index + 1 < len(starts) else len(lines))
        for index, start in enumerate(starts)
    ]

    sections: list[Section] = []
    for index, (start, end) in enumerate(ranges):
        title = _heading_title(lines[start]) or ""
        match = _NUMBER_RE.match(title)
        number = match.group(1) if match else ""
        level = len(number.split(".")) if number else 0
        section_id = f"{doc_id}#{number}" if number else f"{doc_id}#h{index}"
        sections.append(
            Section(
                id=section_id,
                title=title,
                number=number,
                level=level,
                text="".join(lines[start:end]),
            )
        )
    return sections


def segment_by_articles(text: str, *, doc_id: str) -> list[Section]:
    """Split a PRC statute into articles at `第X条` markers (parents = articles).

    Each 第X条 line starts a section running to the next marker (last to EOF).
    Content before the first article is dropped. id = f"{doc_id}#{第X条}".
    """
    lines = text.splitlines(keepends=True)
    starts = [
        index for index, line in enumerate(lines) if _ARTICLE_RE.match(line.strip())
    ]
    sections: list[Section] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        match = _ARTICLE_RE.match(lines[start].strip())
        if match is None:  # guarded by the starts filter
            continue
        number = match.group(1)
        sections.append(
            Section(
                id=f"{doc_id}#{number}",
                title=number,
                number=number,
                level=1,
                text="".join(lines[start:end]),
            )
        )
    return sections


_STRATEGIES = {
    "headings": segment_by_headings,
    "articles": segment_by_articles,
}


def build_parent_child(
    text: str,
    *,
    doc_id: str,
    strategy: str = "headings",
    child_size: int = 500,
    child_overlap: int = 100,
    metadata: dict[str, str] | None = None,
) -> tuple[list[Section], list[Chunk]]:
    """Segment into parents (by strategy), then split each parent's text into
    children via chunk_document, tagging each child's metadata with
    parent_id / section_number / section_title. Return (sections, children)."""
    try:
        segment = _STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown segmentation strategy: {strategy}") from exc

    sections = segment(text, doc_id=doc_id)
    document_metadata = dict(metadata or {})
    children: list[Chunk] = []
    for section in sections:
        section.metadata.update(document_metadata)
        children.extend(
            chunk_document(
                section.text,
                doc_id=section.id,
                size=child_size,
                overlap=child_overlap,
                metadata={
                    **document_metadata,
                    "parent_id": section.id,
                    "section_number": section.number,
                    "section_title": section.title,
                },
            )
        )
    return sections, children


def to_json(parents: list[Section], children: list[Chunk]) -> dict:
    """Serialise the relationships: parents (with their child_ids) and children
    (with parent_id + char span)."""
    children_by_parent: dict[str, list[str]] = {}
    for child in children:
        children_by_parent.setdefault(child.metadata["parent_id"], []).append(child.id)

    return {
        "parents": [
            {
                "id": parent.id,
                "number": parent.number,
                "level": parent.level,
                "title": parent.title,
                "text": parent.text,
                "metadata": parent.metadata,
                "child_ids": children_by_parent.get(parent.id, []),
            }
            for parent in parents
        ],
        "children": [
            {
                "id": child.id,
                "parent_id": child.metadata["parent_id"],
                "char_start": child.metadata["char_start"],
                "char_end": child.metadata["char_end"],
            }
            for child in children
        ],
    }


def main() -> None:
    """Write 等保 parent-child JSON and print 5 sample child -> parent traces."""
    if not DENGBAO.exists():
        print(f"{DENGBAO} not found — run the parsers first.")
        return

    parents, children = build_parent_child(
        DENGBAO.read_text(encoding="utf-8"),
        doc_id="GBT-22239@2019",
    )
    output = PARSED / f"{DENGBAO.stem}.parents.json"
    output.write_text(
        json.dumps(to_json(parents, children), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    numbered = [parent for parent in parents if parent.number]
    print(f"parents={len(parents)} (numbered={len(numbered)}) children={len(children)}")
    print(f"wrote {output}")
    for child in children[:5]:
        print(f"{child.id} -> {child.metadata['parent_id']}")


if __name__ == "__main__":
    main()
