from typing import Optional
from claude_retriever.searcher.types import SearchResult, SearchTool
from anthropic import Anthropic
from elasticsearch import Elasticsearch

import logging
logger = logging.getLogger(__name__)

# Elasticsearch Cloud Searcher

class ElasticsearchCloudSearchTool(SearchTool):

    def __init__(self,
                tool_name: str,
                tool_description: str,
                elasticsearch_cloud_id: str,
                elasticsearch_api_key_id: str,
                elasticsearch_api_key: str,
                elasticsearch_index: str,
                truncate_to_n_tokens: Optional[int] = 5000):
        
        self.index = elasticsearch_index
        self.cloud_id = elasticsearch_cloud_id
        self.api_key_id = elasticsearch_api_key_id
        self.api_key = elasticsearch_api_key
        self._connect_to_elasticsearch()

        self.tool_name = tool_name
        self.tool_description = tool_description
        self.truncate_to_n_tokens = truncate_to_n_tokens
        if truncate_to_n_tokens is not None:
            self.tokenizer = Anthropic().get_tokenizer() 
    
    def _connect_to_elasticsearch(self):
        self.client = Elasticsearch(
            cloud_id=self.cloud_id,
            api_key=(self.api_key_id, self.api_key)
        )
        if not self.client.indices.exists(index=self.index):
            raise ValueError(f"Elasticsearch Index {self.index} does not exist.")
        index_mapping = self.client.indices.get_mapping(index=self.index)
        if "text" not in index_mapping.body[self.index]["mappings"]["properties"].keys():
            raise ValueError(f"Index {self.index} does not have a field called 'text'.")
    
    def truncate_page_content(self, page_content: str) -> str:
        if self.truncate_to_n_tokens is None:
            return page_content.strip()
        else:
            return self.tokenizer.decode(self.tokenizer.encode(page_content).ids[:self.truncate_to_n_tokens]).strip()

    def raw_search(self, query: str, n_search_results_to_use: int) -> list[SearchResult]:

        results = self.client.search(index=self.index,
                                     query={"match": {"text": query}})
        search_results: list[SearchResult] = []
        for result in results["hits"]["hits"]:
            if len(search_results) >= n_search_results_to_use:
                break
            content = result["_source"]["text"]
            search_results.append(SearchResult(source=str(hash(content)), content=content))

        return search_results
    
    def process_raw_search_results(self, results: list[SearchResult]) -> list[list[str]]:
        processed_search_results = [[result.source, self.truncate_page_content(result.content)] for result in results]
        return processed_search_results
    