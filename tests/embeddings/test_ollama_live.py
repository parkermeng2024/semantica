"""Live integration tests for the Ollama bge-m3 embedding path.

These tests hit the real local Ollama server (no mocks). They are skipped
automatically when no server is reachable at OLLAMA_BASE_URL /
http://localhost:11434 or the bge-m3 model is not pulled.
"""

import json
import os
import unittest
import urllib.request

import numpy as np

from semantica.embeddings import TextEmbedder

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_bge_m3_available() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=2) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        return any(m.get("name", "").startswith("bge-m3") for m in tags.get("models", []))
    except Exception:
        return False


@unittest.skipUnless(
    ollama_bge_m3_available(), "Ollama server with bge-m3 not available"
)
class TestOllamaBgeM3Live(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = TextEmbedder(method="ollama", model_name="bge-m3")

    def test_method_and_dimension(self):
        self.assertEqual(self.embedder.get_method(), "ollama")
        self.assertEqual(self.embedder.get_embedding_dimension(), 1024)

    def test_embed_text_normalized(self):
        vec = self.embedder.embed_text("Hello world")
        self.assertEqual(vec.shape, (1024,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_embed_batch_shape(self):
        batch = self.embedder.embed_batch(["a", "b", "c"])
        self.assertEqual(batch.shape, (3, 1024))

    def test_semantic_similarity_ordering(self):
        """bge-m3 is multilingual: cross-lingual paraphrases must outrank unrelated text."""
        anchor, paraphrase, unrelated = self.embedder.embed_batch(
            [
                "知识图谱是一种结构化的语义知识库",
                "A knowledge graph is a structured semantic knowledge base",
                "The recipe calls for two cups of flour",
            ]
        )
        sim_related = float(anchor @ paraphrase)
        sim_unrelated = float(anchor @ unrelated)
        self.assertGreater(sim_related, sim_unrelated)
        self.assertGreater(sim_related, 0.6)

    def test_default_constructor_uses_ollama_bge_m3(self):
        embedder = TextEmbedder()
        self.assertEqual(embedder.get_method(), "ollama")
        self.assertEqual(embedder.model_name, "BAAI/bge-m3")
        self.assertEqual(embedder.get_embedding_dimension(), 1024)


if __name__ == "__main__":
    unittest.main()
