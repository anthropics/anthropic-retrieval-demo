from typing import Optional
from claude_retriever.searcher.types import Embedder, SearchResult, VectorStore, SearchTool
from claude_retriever.searcher.embedders.local import LocalEmbedder
from claude_retriever.constants import DEFAULT_EMBEDDER

import logging
logger = logging.getLogger(__name__)

# Embedding Searcher

class EmbeddingSearchTool(SearchTool):

    def __init__(self, tool_description: str, vector_store: VectorStore, embedder: Optional[Embedder] = None):
        self.tool_description = tool_description
        if embedder is None:
            logger.info(f"Using default embedder: {DEFAULT_EMBEDDER}")
            embedder = LocalEmbedder(DEFAULT_EMBEDDER)
        self.embedder = embedder
        self.vector_store = vector_store

    def raw_search(self, query: str, n_search_results_to_use: int) -> list[SearchResult]:
        query_embedding = self.embedder.embed(query)
        search_results = self.vector_store.query(query_embedding, n_search_results_to_use=n_search_results_to_use)
        return search_results
    
    def process_raw_search_results(self, results: list[SearchResult]) -> list[str]:
        processed_search_results = [result.content for result in results]
        return processed_search_results
    