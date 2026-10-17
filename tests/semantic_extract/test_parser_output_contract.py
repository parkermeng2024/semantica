"""The parse -> normalize -> extract chain must read the parser's own keys.

DocumentParser emits ``{"text", "full_text", "metadata"}`` (see
docs/reference/parse.md). Normalising and extracting used to look only for
``content`` and silently fall back to ``str(record)``, which fed the dict repr
into the pipeline: the text appeared twice (text + full_text), metadata numbers
became entities, and every extra entity minted a spurious relationship.
"""

import unittest

from semantica.normalize import TextNormalizer
from semantica.semantic_extract import NERExtractor, RelationExtractor

TEXT = (
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak and Ronald Wayne in "
    "Cupertino, California."
)
PARSED = {
    "text": TEXT,
    "full_text": TEXT,
    "metadata": {"file_path": "apple.txt", "page_count": 1},
}


class TestParserOutputContract(unittest.TestCase):
    def test_normalize_keeps_record_and_normalizes_its_text_once(self):
        normalized = TextNormalizer().normalize([PARSED])

        self.assertIsInstance(normalized[0], dict)
        self.assertEqual(sorted(normalized[0]), ["full_text", "metadata", "text"])
        self.assertEqual(normalized[0]["metadata"], PARSED["metadata"])
        self.assertEqual(normalized[0]["text"].count("Apple Inc."), 1)

    def test_entities_come_from_the_text_not_the_record_repr(self):
        normalized = TextNormalizer().normalize([PARSED])
        entities = NERExtractor().extract(normalized)
        names = [entity.text for document in entities for entity in document]

        self.assertIn("Apple Inc.", names)
        # The record repr would contribute braces, quotes and metadata keys.
        self.assertFalse(any("{" in name or "metadata" in name for name in names))
        # text and full_text carry the same text: it must not be read twice.
        self.assertEqual(names.count("Apple Inc."), 1)

    def test_relations_are_not_built_from_metadata_digits(self):
        normalized = TextNormalizer().normalize([PARSED])
        entities = NERExtractor().extract(normalized)
        relations = RelationExtractor().extract(normalized, entities)

        triples = [
            (relation.subject.text, relation.predicate, relation.object.text)
            for document in relations
            for relation in document
        ]
        self.assertTrue(triples)
        self.assertFalse(any(a.isdigit() or b.isdigit() for a, _, b in triples))


if __name__ == "__main__":
    unittest.main()
