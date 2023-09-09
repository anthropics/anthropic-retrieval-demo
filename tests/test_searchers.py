import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np

from claude_retriever.searcher.types import SearchResult, Embedder, Embedding
from claude_retriever.searcher.vectorstores.local import LocalVectorStore
from claude_retriever.searcher.searchtools.embeddings import EmbeddingSearchTool

class TestLocalSearcher(unittest.TestCase):

    def setUp(self):

        class DummyEmbedder(Embedder):
            def __init__(self, dim: int = 3):
                self.dim = dim
            def embed(self, text: str) -> Embedding:
                emb = self.embed_batch([text])
                return emb[0]
            def embed_batch(self, texts: list[str]) -> list[Embedding]:
                embeddings = np.random.rand(len(texts), self.dim).tolist()
                return [Embedding(embedding=embedding, text=text) for embedding, text in zip(embeddings, texts)]
            def tokenizer_encode(self, text: str) -> list[int]:
                return [1, 2, 3]
            def tokenizer_decode(self, tokens: list[int]) -> str:
                return "This is a test sentence."
    
        self.embedder = DummyEmbedder()
        self.vector_store = LocalVectorStore("tests/data/local_db.jsonl")
        self.searchtool = EmbeddingSearchTool(tool_description="This is a test search tool.",
                                              vector_store=self.vector_store,
                                              embedder=self.embedder)

    def test_search(self):
        query = "This is a test sentence."
        n_search_results_to_use = 2
        search_results = self.searchtool.raw_search(query, n_search_results_to_use)
        self.assertIsInstance(search_results, list)
        self.assertEqual(len(search_results), n_search_results_to_use)
        for result in search_results:
            self.assertIsInstance(result, SearchResult)
            self.assertIsInstance(result.content, str)

if __name__ == "__main__":
    unittest.main()
