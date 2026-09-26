import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Import the module to be tested
from semantica.embeddings.text_embedder import TextEmbedder
from semantica.utils.exceptions import ProcessingError

class TestTextEmbedder(unittest.TestCase):
    
    def setUp(self):
        # Create a mock for sentence_transformers.SentenceTransformer
        self.mock_st_patcher = patch('semantica.embeddings.text_embedder.SentenceTransformer')
        self.mock_st_class = self.mock_st_patcher.start()
        
        # Create a mock for fastembed.TextEmbedding
        self.mock_fe_patcher = patch('semantica.embeddings.text_embedder.TextEmbedding')
        self.mock_fe_class = self.mock_fe_patcher.start()
        
        # Patch availability flags
        self.st_avail_patcher = patch('semantica.embeddings.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE', True)
        self.st_avail_patcher.start()
        
        self.fe_avail_patcher = patch('semantica.embeddings.text_embedder.FASTEMBED_AVAILABLE', True)
        self.fe_avail_patcher.start()

    def tearDown(self):
        self.mock_st_patcher.stop()
        self.mock_fe_patcher.stop()
        self.st_avail_patcher.stop()
        self.fe_avail_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters (ollama)."""
        with patch(
            "urllib.request.urlopen",
            side_effect=TestOllamaEmbedder()._mock_urlopen([[0.1] * 1024]),
        ):
            embedder = TextEmbedder()
        self.assertEqual(embedder.method, "ollama")
        self.assertEqual(embedder.model_name, "BAAI/bge-m3")
        self.assertIsNotNone(embedder.ollama_base_url)
        self.assertIsNone(embedder.fastembed_model)
        self.assertIsNone(embedder.model)

    def test_init_sentence_transformers(self):
        """Test initialization with sentence-transformers method."""
        embedder = TextEmbedder(method="sentence_transformers")
        self.assertEqual(embedder.method, "sentence_transformers")
        self.mock_st_class.assert_called_once()
        self.assertIsNotNone(embedder.model)
        self.assertIsNone(embedder.fastembed_model)

    def test_init_fastembed(self):
        """Test initialization with fastembed method."""
        embedder = TextEmbedder(method="fastembed")
        self.assertEqual(embedder.method, "fastembed")
        self.mock_fe_class.assert_called_once()
        self.assertIsNotNone(embedder.fastembed_model)
        self.assertIsNone(embedder.model)

    def test_embed_text_sentence_transformers(self):
        """Test embedding generation with sentence-transformers."""
        embedder = TextEmbedder(method="sentence_transformers")
        
        # Mock the encode method
        mock_embedding = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        embedder.model.encode.return_value = mock_embedding
        
        result = embedder.embed_text("test text")
        
        self.assertTrue(np.array_equal(result, mock_embedding[0]))
        embedder.model.encode.assert_called_with(["test text"], normalize_embeddings=True)

    def test_embed_text_fastembed(self):
        """Test embedding generation with fastembed."""
        embedder = TextEmbedder(method="fastembed")
        
        # Mock the embed method
        mock_embedding = [0.1, 0.2, 0.3]
        # FastEmbed returns a generator of embeddings
        embedder.fastembed_model.embed.return_value = iter([mock_embedding])
        
        result = embedder.embed_text("test text", normalize=False)
        
        # Note: TextEmbedder.embed_text normalizes manually for FastEmbed if self.normalize is True
        # Default is True. The mock result [0.1, 0.2, 0.3] will be normalized.
        expected_norm = np.linalg.norm(np.array(mock_embedding, dtype=np.float32))
        expected = np.array(mock_embedding, dtype=np.float32) / expected_norm
        
        self.assertTrue(np.allclose(result, expected))
        embedder.fastembed_model.embed.assert_called_with(["test text"])

    def test_embed_text_empty(self):
        """Test error handling for empty text."""
        embedder = TextEmbedder()
        with self.assertRaises(ProcessingError):
            embedder.embed_text("")
        with self.assertRaises(ProcessingError):
            embedder.embed_text("   ")

    def test_embed_batch_sentence_transformers(self):
        """Test batch embedding with sentence-transformers."""
        embedder = TextEmbedder(method="sentence_transformers")
        
        # Mock the encode method
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        texts = ["text1", "text2"]
        results = embedder.embed_batch(texts)
        
        self.assertTrue(np.array_equal(results, mock_embeddings))
        embedder.model.encode.assert_called_with(texts, normalize_embeddings=True)

    def test_embed_batch_fastembed(self):
        """Test batch embedding with fastembed."""
        embedder = TextEmbedder(method="fastembed")
        
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        embedder.fastembed_model.embed.return_value = iter(mock_embeddings)
        
        texts = ["text1", "text2"]
        results = embedder.embed_batch(texts)
        
        # Should be normalized manually
        expected = np.array(mock_embeddings, dtype=np.float32)
        norms = np.linalg.norm(expected, axis=1, keepdims=True)
        expected = expected / norms
        
        self.assertTrue(np.allclose(results, expected))

    def test_fallback_method(self):
        """Test fallback method when libraries are unavailable."""
        # Unpatch availability to simulate missing libraries
        self.st_avail_patcher.stop()
        self.fe_avail_patcher.stop()

        with patch('semantica.embeddings.text_embedder.SENTENCE_TRANSFORMERS_AVAILABLE', False), \
             patch('semantica.embeddings.text_embedder.FASTEMBED_AVAILABLE', False), \
             patch('urllib.request.urlopen', side_effect=ConnectionRefusedError("down")):

            embedder = TextEmbedder()
            self.assertIsNone(embedder.model)
            self.assertIsNone(embedder.fastembed_model)
            self.assertIsNone(embedder.ollama_base_url)
            
            # Should use fallback (hashing)
            result = embedder.embed_text("test")
            self.assertIsInstance(result, np.ndarray)
            # Check length is 128 (as per fallback implementation)
            self.assertTrue(len(result) <= 128) 
            
            # Batch fallback
            results = embedder.embed_batch(["t1", "t2"])
            self.assertEqual(len(results), 2)

    def test_set_model(self):
        """Test dynamic model switching."""
        with patch(
            "urllib.request.urlopen",
            side_effect=TestOllamaEmbedder()._mock_urlopen([[0.1] * 1024]),
        ):
            embedder = TextEmbedder()  # Default Ollama
        self.assertEqual(embedder.method, "ollama")

        embedder.set_model(method="sentence_transformers", model_name="new-model")
        self.assertEqual(embedder.method, "sentence_transformers")
        self.assertEqual(embedder.model_name, "new-model")
        self.mock_st_class.assert_called()

class TestOllamaEmbedder(unittest.TestCase):
    """Tests for the Ollama embedding method (HTTP calls mocked)."""

    def _mock_urlopen(self, embeddings):
        """Build a urlopen replacement returning the given embeddings as JSON."""
        import json

        def fake_urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            n = len(payload["input"])
            body = json.dumps({"embeddings": embeddings[:n]}).encode("utf-8")
            response = MagicMock()
            response.read.return_value = body
            response.__enter__ = lambda s: s
            response.__exit__ = MagicMock(return_value=False)
            return response

        return fake_urlopen

    def test_init_and_embed(self):
        """Ollama init probes the model; embed_text returns a normalized vector."""
        with patch(
            "urllib.request.urlopen",
            side_effect=self._mock_urlopen([[3.0, 4.0, 0.0]]),
        ):
            embedder = TextEmbedder(method="ollama", model_name="bge-m3")
            self.assertEqual(embedder.get_method(), "ollama")
            self.assertEqual(embedder.get_embedding_dimension(), 3)

            result = embedder.embed_text("hello")
            self.assertIsInstance(result, np.ndarray)
            # [3, 4, 0] normalized -> [0.6, 0.8, 0]
            self.assertTrue(np.allclose(result, [0.6, 0.8, 0.0]))

    def test_hub_prefix_stripped(self):
        """Hub-prefixed names like BAAI/bge-m3 map to the bare Ollama name."""
        import json
        import urllib.request

        seen = {}

        def fake_urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            seen["model"] = payload["model"]
            body = json.dumps(
                {"embeddings": [[0.1] * 4 for _ in payload["input"]]}
            ).encode("utf-8")
            response = MagicMock()
            response.read.return_value = body
            response.__enter__ = lambda s: s
            response.__exit__ = MagicMock(return_value=False)
            return response

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            embedder = TextEmbedder(method="ollama", model_name="BAAI/bge-m3")
        self.assertEqual(seen["model"], "bge-m3")
        self.assertEqual(embedder.get_embedding_dimension(), 4)

    def test_embed_batch(self):
        """embed_batch sends all texts in one request and normalizes rows."""
        with patch(
            "urllib.request.urlopen",
            side_effect=self._mock_urlopen([[1.0, 0.0], [0.0, 2.0]]),
        ):
            embedder = TextEmbedder(method="ollama", model_name="bge-m3")
            results = embedder.embed_batch(["a", "b"])
        self.assertEqual(results.shape, (2, 2))
        self.assertTrue(np.allclose(results, [[1.0, 0.0], [0.0, 1.0]]))

    def test_fallback_when_server_down(self):
        """Unreachable Ollama server falls back to hash-based embeddings."""
        with patch(
            "urllib.request.urlopen", side_effect=ConnectionRefusedError("down")
        ):
            embedder = TextEmbedder(method="ollama", model_name="bge-m3")
            self.assertIsNone(embedder.ollama_base_url)
            self.assertEqual(embedder.get_method(), "fallback")
            result = embedder.embed_text("hello")
            self.assertIsInstance(result, np.ndarray)

if __name__ == '__main__':
    unittest.main()
