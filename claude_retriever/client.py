from typing import Optional, Tuple
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from claude_retriever.prompts import citations_prompt, retrieval_prompt
from .searcher.types import SearchTool, SearchResult, Tool
import logging
import re
from utils import format_results_full

logger = logging.getLogger(__name__)

SIMPLE_ANSWER_PROMPT = """
{results} Using the search results provided within the <search_results></search_results> tags, please answer the following query <query>{query}</query>.
"""

class ClientWithRetrieval(Anthropic):

    def __init__(self, search_tool: Optional[SearchTool] = None, verbose: bool = True, *args, **kwargs):
        """
        Initializes the ClientWithRetrieval class.
        
        Parameters:
            search_tool (SearchTool): SearchTool object to handle searching
            verbose (bool): Whether to print verbose logging
            *args, **kwargs: Passed to superclass init
        """
        super().__init__(*args, **kwargs)
        self.search_tool = search_tool
        self.verbose = verbose
    
    def retrieve(self,
                       query: str,
                       model: str,
                       n_search_results_to_use: int = 3,
                       stop_sequences: list[str] = [HUMAN_PROMPT],
                       max_tokens_to_sample: int = 1000,
                       max_searches_to_try: int = 5,
                       temperature: float = 1.0) -> list[SearchResult]:
        """
        Main method to retrieve relevant search results for a query with a provided search tool.
        
        Constructs RETRIEVAL prompt with query and search tool description. 
        Keeps sampling Claude completions until stop sequence hit.
        Extracts search results and accumulates all raw results.
        
        Returns:
            list[SearchResult]: List of all raw search results
        """
        assert self.search_tool is not None, "SearchTool must be provided to use .retrieve()"

        prompt = retrieval_prompt(self.search_tool.tool_name, self.search_tool.tool_description, query)
        token_budget = max_tokens_to_sample
        all_raw_search_results: list[SearchResult] = []
        for tries in range(max_searches_to_try):
            partial_completion = self.completions.create(prompt = prompt,
                                                     stop_sequences=stop_sequences + ['</function_calls>'],
                                                     model=model,
                                                     max_tokens_to_sample = token_budget,
                                                     temperature = temperature)
            partial_completion, stop_reason, stop_seq = partial_completion.completion, partial_completion.stop_reason, partial_completion.stop # type: ignore
            logger.info(partial_completion)
            token_budget -= self.count_tokens(partial_completion)
            prompt += partial_completion
            if stop_reason == 'stop_sequence' and stop_seq == '</function_calls>':
                logger.info(f'Attempting search number {tries}.')
                raw_search_results, formatted_search_results = self._search_query_stop(partial_completion, n_search_results_to_use)
                prompt += '</function_calls>' + formatted_search_results
                all_raw_search_results += raw_search_results
            else:
                break
        return all_raw_search_results
    
    def answer_with_results(self, raw_search_results: list[str]|list[SearchResult], query: str, model: str, temperature: float, format_results: bool =False):
        """Generates an RAG response based on search results and a query. If format_results is True,
           formats the raw search results first. Set format_results to True if you are using this method standalone without retrieve().
        
        Returns:
            str: Claude's answer to the query
        """

        if isinstance(raw_search_results[0], str):
            raw_search_results  = [SearchResult(content=s, source=str(hash(s))) for s in raw_search_results] # type: ignore
        
        processed_search_results = [[result.source, result.content.strip()] for result in raw_search_results] # type: ignore
        formatted_search_results = format_results_full(processed_search_results)

        # Use the SIMPLE_ANSWER_PROMPT if you do not want citations in your answer
        # prompt = f"{HUMAN_PROMPT} {SIMPLE_ANSWER_PROMPT.format(query=query, results=formatted_search_results)}{AI_PROMPT}"
        
        prompt = citations_prompt(formatted_search_results, query)

        answer = self.completions.create(
            prompt=prompt, 
            model=model, 
            temperature=temperature, 
            max_tokens_to_sample=1000
        ).completion
        
        return answer
    
    def completion_with_retrieval(self,
                                        query: str,
                                        model: str,
                                        n_search_results_to_use: int = 3,
                                        stop_sequences: list[str] = [HUMAN_PROMPT],
                                        max_tokens_to_sample: int = 1000,
                                        max_searches_to_try: int = 5,
                                        temperature: float = 1.0) -> str:
        """
        Gets a final completion from retrieval results        
        
        Calls retrieve() to get search results.
        Calls answer_with_results() with search results and query.
        
        Returns:
            str: Claude's answer to the query
        """
        search_results = self.retrieve(query, model=model,
                                                 n_search_results_to_use=n_search_results_to_use, stop_sequences=stop_sequences,
                                                 max_tokens_to_sample=max_tokens_to_sample,
                                                 max_searches_to_try=max_searches_to_try,
                                                 temperature=temperature)
        answer = self.answer_with_results(search_results, query, model, temperature)
        return answer
    
    # Helper methods
    def _search_query_stop(self, partial_completion: str, n_search_results_to_use: int) -> Tuple[list[SearchResult], str]:
        """
        Helper to handle search query stop case.
        
        Extracts search query from completion text.
        Runs search using SearchTool. 
        Formats search results.
        
        Returns:
            tuple: 
                list[SearchResult]: Raw search results
                str: Formatted search result text
        """
        assert self.search_tool is not None, "SearchTool was not provided for client"
        search_query = self.extract_between_tags('query', partial_completion + '</query>') 
        if search_query is None:
            raise Exception(f'Completion with retrieval failed as partial completion returned mismatched <query> tags.')
        if self.verbose:
            logger.info('\n'+'-'*20 + f'\nPausing stream because Claude has issued a query in <query> tags: <query>{search_query}</query>\n' + '-'*20)
        logger.info(f'Running search query against SearchTool: {search_query}')
        search_results = self.search_tool.raw_search(search_query, n_search_results_to_use)
        extracted_search_results = self.search_tool.process_raw_search_results(search_results)
        formatted_search_results = format_results_full(extracted_search_results)

        if self.verbose:
            logger.info('\n' + '-'*20 + f'\nThe SearchTool has returned the following search results:\n\n{formatted_search_results}\n\n' + '-'*20 + '\n')
        return search_results, formatted_search_results
    
    def extract_between_tags(self, tag, string, strip=True):
        """
        Helper to extract text between XML tags.
        
        Finds last match of specified tags in string.
        Handles edge cases and stripping.
        
        Returns:
            str: Extracted string between tags
        """
        ext_list = re.findall(f"<{tag}\\s?>(.+?)</{tag}\\s?>", string, re.DOTALL)
        if strip:
            ext_list = [e.strip() for e in ext_list]
        
        if ext_list:
            return ext_list[-1]
        else:
            return None
