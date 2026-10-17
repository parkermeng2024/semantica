"""Tests for semantica.utils.documents.document_text."""

import unittest

from semantica.utils.documents import document_text


class TestDocumentText(unittest.TestCase):
    def test_parser_shape_prefers_text_over_full_text(self):
        record = {"text": "short", "full_text": "short", "metadata": {"page_count": 1}}
        self.assertEqual(document_text(record), "short")

    def test_content_wins_when_present(self):
        self.assertEqual(document_text({"content": "c", "text": "t"}), "c")

    def test_plain_string_passes_through(self):
        self.assertEqual(document_text("hello"), "hello")

    def test_bytes_attribute_is_decoded(self):
        class Doc:
            content = "苹果公司".encode("utf-8")

        self.assertEqual(document_text(Doc()), "苹果公司")

    def test_unrecognised_payload_falls_back_to_str(self):
        self.assertEqual(document_text(42), "42")
        self.assertEqual(document_text({}), "{}")


if __name__ == "__main__":
    unittest.main()
