import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import tempfile
from scipy.sparse import csr_matrix
import numpy as np

from claude_retriever.searcher.vectorstores.local import LocalVectorStore, LocalHybridVectorStore
from claude_retriever.searcher.types import Embedding, SparseEmbeddingData, HybridEmbedding

class TestLocalVectorStore(unittest.TestCase):

    def test_load_jsonl(self):
        vector_store = LocalVectorStore("tests/data/local_db.jsonl")
        self.assertEqual(len(vector_store.embeddings), 3)
        self.assertEqual(vector_store.embeddings[0].text, "text1")
        self.assertEqual(vector_store.embeddings[1].text, "text2")
        self.assertEqual(vector_store.embeddings[2].text, "text3")

    def test_query(self):
        vector_store = LocalVectorStore("tests/data/local_db.jsonl")
        query_embedding = Embedding([1, 2, 3], "query")
        n_search_results_to_use_results = vector_store.query(query_embedding, n_search_results_to_use=2)
        self.assertEqual(len(n_search_results_to_use_results), 2)
        self.assertEqual(n_search_results_to_use_results[0].content, "text1")
        self.assertEqual(n_search_results_to_use_results[1].content, "text2")

class TestLocalHybridVectorStore(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self.temp_file.close()
        self.disk_path = self.temp_file.name

        self.dense_embedding1 = [0.5, 0.5, 0.5, 0.5]
        self.dense_embedding2 = [0.0, 1.0, 0.0, 1.0]
        self.dense_embedding3 = [0.25, 0.75, 0.25, 0.75]

        self.sparse_embedding1 = SparseEmbeddingData([0, 1], [1.0, 1.0], 3)
        self.sparse_embedding2 = SparseEmbeddingData([2, 3], [1.0, 1.0], 3)
        self.sparse_embedding3 = SparseEmbeddingData([1], [0.5], 3)

        self.hybrid_embedding1 = HybridEmbedding(embedding=self.dense_embedding1, sparse_embedding=self.sparse_embedding1, text="Hello")
        self.hybrid_embedding2 = HybridEmbedding(embedding=self.dense_embedding2, sparse_embedding=self.sparse_embedding2, text="World")
        self.hybrid_embedding3 = HybridEmbedding(embedding=self.dense_embedding3, sparse_embedding=self.sparse_embedding3, text="Pandas are cute")

        self.hybrid_embedding4 = HybridEmbedding(embedding=self.dense_embedding1, sparse_embedding=self.sparse_embedding2, text="Hello World")

    def tearDown(self):
        os.remove(self.disk_path)

    def test_init(self):
        vector_store = LocalHybridVectorStore(self.disk_path)
        self.assertEqual(vector_store.disk_path, self.disk_path)
        self.assertEqual(len(vector_store.embeddings), 0)

    def test_upsert_and_load_data(self):
        vector_store = LocalHybridVectorStore(self.disk_path)
        vector_store.upsert([self.hybrid_embedding1, self.hybrid_embedding2, self.hybrid_embedding3])
        self.assertEqual(len(vector_store.embeddings), 3)

    def test_sparse_embedding_data_to_sparse_matrix(self):
        sparse_matrix = LocalHybridVectorStore.sparse_embedding_data_to_sparse_matrix([self.sparse_embedding1, self.sparse_embedding2])
        expected_matrix = csr_matrix(([1.0, 1.0, 1.0, 1.0], ([0, 0, 1, 1], [0, 1, 2, 3])), shape=(2, 4))
        self.assertTrue(np.array_equal(sparse_matrix.toarray(), expected_matrix.toarray()))

    def test_query(self):
        vector_store = LocalHybridVectorStore(self.disk_path)
        vector_store.upsert([self.hybrid_embedding1, self.hybrid_embedding2, self.hybrid_embedding3])

        query_embedding = self.hybrid_embedding4

        results = vector_store.query(query_embedding, 2, query_strategy="split_return")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].content, "Hello")
        self.assertEqual(results[1].content, "World")

        results = vector_store.query(query_embedding, 2, query_strategy="weighted", sparse_weight=0.9)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].content, "World")
        self.assertEqual(results[1].content, "Hello")

    def test_query_empty_store(self):
        vector_store = LocalHybridVectorStore(self.disk_path)
        query_embedding = HybridEmbedding(embedding=[0.25, 0.75, 0.25, 0.75], sparse_embedding=SparseEmbeddingData([1, 2], [0.5, 0.5], 3), text="Query")

        results = vector_store.query(query_embedding, 2, query_strategy="split_return")
        self.assertEqual(len(results), 0)

        results = vector_store.query(query_embedding, 2, query_strategy="weighted")
        self.assertEqual(len(results), 0)

    def test_query_with_different_strategies(self):
        vector_store = LocalHybridVectorStore(self.disk_path)
        vector_store.upsert([self.hybrid_embedding1, self.hybrid_embedding2])

        query_embedding = HybridEmbedding(embedding=[0.25, 0.75, 0.25, 0.75], sparse_embedding=SparseEmbeddingData([1, 2], [0.5, 0.5], 3), text="Query")

        results_split_return = vector_store.query(query_embedding, 2, query_strategy="split_return")
        results_weighted = vector_store.query(query_embedding, 2, query_strategy="weighted")
        self.assertEqual(len(results_split_return), len(results_weighted))
        self.assertEqual(results_split_return[0].content, results_weighted[0].content)
        self.assertEqual(results_split_return[1].content, results_weighted[1].content)

    def test_query_with_invalid_strategy(self):
        vector_store = LocalHybridVectorStore(self.disk_path)
        vector_store.upsert([self.hybrid_embedding1, self.hybrid_embedding2])

        query_embedding = self.hybrid_embedding4

        results = vector_store.query(query_embedding, 2, query_strategy="invalid_strategy")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].content, "Hello")

if __name__ == '__main__':
    unittest.main()