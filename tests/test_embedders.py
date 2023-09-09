import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from claude_retriever.searcher.types import Embedding
from claude_retriever.searcher.embedders.local import LocalEmbedder
from claude_retriever.searcher.embedders.huggingface import HuggingFaceEmbedder

import dotenv
dotenv.load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

class TestLocalEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = LocalEmbedder(model_name=DEFAULT_EMBEDDING_MODEL)

    def test_embed(self):
        query = "This is a test sentence."
        result = self.embedder.embed(query)
        self.assertIsInstance(result, Embedding)
        self.assertEqual(result.text, query)
        self.assertIsInstance(result.embedding, list)

    def test_embed_batch(self):
        queries = ["This is a test sentence.", "Another test sentence."]
        results = self.embedder.embed_batch(queries)
        self.assertIsInstance(results, list)
        for result, query in zip(results, queries):
            self.assertIsInstance(result, Embedding)
            self.assertEqual(result.text, query)
            self.assertIsInstance(result.embedding, list)

class TestHuggingFaceEmbedder(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        assert self.api_key is not None, "Huggingface API key is not set."
        self.embedder = HuggingFaceEmbedder(api_key=self.api_key, model_name=DEFAULT_EMBEDDING_MODEL)

    def test_embed(self):
        query = "This is a test sentence."
        result = self.embedder.embed(query)
        self.assertIsInstance(result, Embedding)
        self.assertEqual(result.text, query)
        self.assertIsInstance(result.embedding, list)

    def test_embed_batch(self):
        queries = ["This is a test sentence.", "Another test sentence."]
        results = self.embedder.embed_batch(queries)
        self.assertIsInstance(results, list)
        for result, query in zip(results, queries):
            self.assertIsInstance(result, Embedding)
            self.assertEqual(result.text, query)
            self.assertIsInstance(result.embedding, list)

if __name__ == "__main__":
    unittest.main()
