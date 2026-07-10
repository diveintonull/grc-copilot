"""Versioned regulation metadata shared by ingestion and retrieval.

The schema keeps source identity separate from parsing and chunking.  A clause
ID includes the document version so revised regulations cannot silently replace
or merge with older clauses in the vector store.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def content_sha256(text: str) -> str:
    """Return a deterministic SHA-256 digest for UTF-8 document content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_identifier_part(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    if "@" in value or "#" in value:
        raise ValueError(f"{name} must not contain '@' or '#'")


@dataclass(frozen=True, slots=True)
class DocumentMeta:
    source_id: str
    title: str
    jurisdiction: str
    version: str
    effective_date: date | None
    source_url: str
    content_hash: str

    def __post_init__(self) -> None:
        _require_identifier_part("source_id", self.source_id)
        _require_identifier_part("version", self.version)
        for name in ("title", "jurisdiction", "source_url"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not _SHA256_RE.fullmatch(self.content_hash):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")

    @property
    def document_id(self) -> str:
        return f"{self.source_id}@{self.version}"

    def to_payload(self) -> dict[str, str]:
        """Return JSON-safe provenance fields for a Qdrant payload."""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "jurisdiction": self.jurisdiction,
            "version": self.version,
            "effective_date": self.effective_date.isoformat() if self.effective_date else "",
            "source_url": self.source_url,
            "content_hash": self.content_hash,
        }


def make_parent_id(document: DocumentMeta, section_number: str) -> str:
    """Build a version-aware clause ID such as ``law@2025#第二十一条``."""
    if not section_number.strip():
        raise ValueError("section_number must not be empty")
    if "#" in section_number:
        raise ValueError("section_number must not contain '#'")
    return f"{document.document_id}#{section_number}"


@dataclass(frozen=True, slots=True)
class ParentSection:
    id: str
    document: DocumentMeta
    number: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    parent_id: str
    text: str
    char_start: int
    char_end: int
    metadata: dict[str, str] = field(default_factory=dict)


def validate_parent_links(parents: list[ParentSection], children: list[Chunk]) -> None:
    """Reject child chunks whose ``parent_id`` is absent from the parent set."""
    parent_ids = {parent.id for parent in parents}
    missing = sorted({child.parent_id for child in children if child.parent_id not in parent_ids})
    if missing:
        raise ValueError(f"children reference missing parents: {', '.join(missing)}")
