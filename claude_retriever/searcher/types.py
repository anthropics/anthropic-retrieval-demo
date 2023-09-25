from dataclasses import dataclass
from abc import ABC, abstractmethod

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir) 

#########################################################
## Embedder: Convert texts to embeddings
#########################################################

@dataclass
class Embedding:
    """
    An embedding of a text, along with the text itself and any metadata associated with it.
    """
    embedding: list[float]
    text: str

@dataclass
class SparseEmbeddingData:
    """
    A sparse embedding index, containing a list of indices and a list of values.
    """
    indices: list[int]
    values: list[float]
    max_index: int

@dataclass
class HybridEmbedding(Embedding):
    """
    A hybrid embedding, containing a dense embedding and a sparse embedding index.
    """
    sparse_embedding: SparseEmbeddingData

class Embedder(ABC):
    """
    An embedder that can embed a single text or a batch of texts.
    """
    dim: int
    
    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """
        Embeds a single text.

        :param text: The text to embed.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """
        Embeds a batch of texts.

        :param texts: The texts to embed.
        """
        raise NotImplementedError()


#########################################################
## Search Tool: A wrapper around a searcher with instructions and formatting to help Claude use it
#########################################################

@dataclass
class SearchResult:
    """
    A single search result.
    """
    content: str
    source: str

class Tool(ABC):
    tool_name: str
    tool_description: str

class SearchTool(Tool):
    """
    A search tool that can run a query and return a formatted string of search results.
    """

    def __init__(self, tool_name: str, tool_description: str):
        self.tool_name = tool_name
        self.tool_description = tool_description

    @abstractmethod
    def raw_search(self, query: str, n_search_results_to_use: int) -> list[SearchResult]:
        """
        Runs a query using the searcher, then returns the raw search results without formatting.

        :param query: The query to run.
        :param n_search_results_to_use: The number of results to return.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def process_raw_search_results(
        self, results: list[SearchResult],
    ) -> list[list[str]]:
        """
        Extracts the raw search content from the search results and returns a list of strings that can be passed to Claude.

        :param results: The search results to extract.
        """
        raise NotImplementedError()
    
    def search(self, query: str, n_search_results_to_use: int) -> str:
        from utils import format_results_full # Avoids circular import

        raw_search_results = self.raw_search(query, n_search_results_to_use)
        processed_search_results = self.process_raw_search_results(raw_search_results)
        displayable_search_results = format_results_full(processed_search_results)
        return displayable_search_results 



#########################################################
## VectorStore: An interface to a vector store that can upsert embeddings and run queries
#########################################################

class VectorStore(ABC):
    """
    An interface to a vector store that can upsert embeddings and run queries.
    """
    
    @abstractmethod
    def upsert(self, embeddings: list[Embedding]) -> None:
        """
        Upserts a list of embeddings into the vector store.

        :param embeddings: The embeddings to upsert.
        """
        raise NotImplementedError()

    @abstractmethod
    def query(self, query_embedding: Embedding, n_search_results_to_use: int = 10) -> list[SearchResult]:
        """
        Runs a query using the vector store and returns the results.

        :param query_embedding: The embedding to query with.
        :param n_search_results_to_use: The number of results to return.
        """
        raise NotImplementedError()