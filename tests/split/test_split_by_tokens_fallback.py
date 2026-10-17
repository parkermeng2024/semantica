"""split_by_tokens must decode with the backend that produced the tokens."""

import unittest
from unittest.mock import patch

from semantica.split.methods import split_by_tokens


class TestSplitByTokensFallback(unittest.TestCase):
    def test_transformers_load_failure_falls_back_to_words(self):
        # tiktoken missing plus an unloadable HuggingFace tokenizer (offline or
        # air-gapped install) used to leave `tokenizer_obj` unbound while the
        # decode step still expected it, raising UnboundLocalError.
        with (
            patch("semantica.split.methods.TIKTOKEN_AVAILABLE", False),
            patch("semantica.split.methods.TRANSFORMERS_AVAILABLE", True),
            patch("semantica.split.methods.AutoTokenizer") as auto_tokenizer,
        ):
            auto_tokenizer.from_pretrained.side_effect = OSError("offline")
            chunks = split_by_tokens("word " * 6, chunk_size=50, chunk_overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text.split(), ["word"] * 6)
        self.assertEqual(chunks[0].metadata["token_count"], 6)

    def test_without_any_tokenizer_backend_uses_words(self):
        with (
            patch("semantica.split.methods.TIKTOKEN_AVAILABLE", False),
            patch("semantica.split.methods.TRANSFORMERS_AVAILABLE", False),
        ):
            chunks = split_by_tokens("word " * 6, chunk_size=50, chunk_overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text.split(), ["word"] * 6)


if __name__ == "__main__":
    unittest.main()
