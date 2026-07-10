"""Contract tests for versioned regulation metadata and stable IDs."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import date

from ingest.schema import (
    Chunk,
    DocumentMeta,
    ParentSection,
    content_sha256,
    make_parent_id,
    validate_parent_links,
)


def document(version: str = "2019") -> DocumentMeta:
    return DocumentMeta(
        source_id="GBT-22239",
        title="网络安全等级保护基本要求",
        jurisdiction="CN",
        version=version,
        effective_date=date(2019, 12, 1),
        source_url="https://openstd.samr.gov.cn/",
        content_hash=content_sha256("标准正文"),
    )


class DocumentMetaTests(unittest.TestCase):
    def test_same_source_different_versions_have_different_parent_ids(self) -> None:
        old_id = make_parent_id(document("2019"), "6.1.1")
        new_id = make_parent_id(document("2025"), "6.1.1")

        self.assertEqual(old_id, "GBT-22239@2019#6.1.1")
        self.assertEqual(new_id, "GBT-22239@2025#6.1.1")
        self.assertNotEqual(old_id, new_id)

    def test_missing_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "version"):
            document("")

    def test_content_hash_is_stable_and_content_sensitive(self) -> None:
        self.assertEqual(content_sha256("同一正文"), content_sha256("同一正文"))
        self.assertNotEqual(content_sha256("正文 A"), content_sha256("正文 B"))
        self.assertEqual(len(content_sha256("正文")), 64)

    def test_metadata_is_immutable_and_serializes_for_qdrant(self) -> None:
        meta = document()

        with self.assertRaises(FrozenInstanceError):
            meta.version = "2025"  # type: ignore[misc]

        self.assertEqual(
            meta.to_payload(),
            {
                "source_id": "GBT-22239",
                "title": "网络安全等级保护基本要求",
                "jurisdiction": "CN",
                "version": "2019",
                "effective_date": "2019-12-01",
                "source_url": "https://openstd.samr.gov.cn/",
                "content_hash": content_sha256("标准正文"),
            },
        )


class ParentChildContractTests(unittest.TestCase):
    def test_child_cannot_reference_a_missing_parent(self) -> None:
        meta = document()
        parent = ParentSection(
            id=make_parent_id(meta, "6.1.1"),
            document=meta,
            number="6.1.1",
            title="安全物理环境",
            text="条款正文",
        )
        orphan = Chunk(
            id="GBT-22239@2019#6.1.2:0",
            parent_id="GBT-22239@2019#6.1.2",
            text="孤立子块",
            char_start=0,
            char_end=4,
        )

        with self.assertRaisesRegex(ValueError, orphan.parent_id):
            validate_parent_links([parent], [orphan])

    def test_valid_parent_child_relationship_passes(self) -> None:
        meta = document()
        parent_id = make_parent_id(meta, "6.1.1")
        parent = ParentSection(
            id=parent_id,
            document=meta,
            number="6.1.1",
            title="安全物理环境",
            text="条款正文",
        )
        child = Chunk(
            id=f"{parent_id}:0",
            parent_id=parent_id,
            text="条款正文",
            char_start=0,
            char_end=4,
        )

        validate_parent_links([parent], [child])


if __name__ == "__main__":
    unittest.main()
