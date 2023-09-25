import os
from typing import Optional
from claude_retriever.searcher.types import SearchResult, SearchTool
from claude_retriever.utils import scrape_url
from dataclasses import dataclass
import requests
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt

import logging
logger = logging.getLogger(__name__)

# Brave Searcher

BRAVE_DESCRIPTION = """The search engine will search using the Brave search engine for web pages with keywords similar to your query. It returns for each page its title, a summary and potentially the full page content. Use this tool if you want to get up-to-date and comprehensive information on a topic."""

class BraveAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def search(self, query: str) -> dict:
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query,
                    "count": 20 # Max number of results to return, can filter down later
                    },
            headers=headers,
            timeout=60
        )
        if resp.status_code != 200:
            logger.error(f"Search request failed: {resp.text}")
            return {}
        return resp.json()

class BraveSearchTool(SearchTool):

    def __init__(self, brave_api_key: str,
                 tool_name: str = "Brave Search Engine",
                 tool_description: str = BRAVE_DESCRIPTION,
                 summarize_with_claude: bool = False,
                 anthropic_api_key: Optional[str] = None):
        """
        :param brave_api_key: The Brave API key to use for searching.
        :param tool_description: The description of the tool.
        :param summarize_with_claude: Whether to summarize the scraped web pages with Claude.
        :param anthropic_api_key: The anthropic API key to use for summarizing with Claude.
        """

        self.api = BraveAPI(brave_api_key)
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.summarize_with_claude = summarize_with_claude
        if summarize_with_claude and anthropic_api_key is None:
            try:
                anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
            except KeyError:
                raise ValueError("If you want to summarize with Claude, you must provide an anthropic_api_key.")
        self.anthropic_api_key = anthropic_api_key

    def parse_faq(self, faq: dict) -> SearchResult:
        """
        https://api.search.brave.com/app/documentation/responses#FAQ
        """
        snippet = f"""FAQ Title: {faq.get('title', "Unknown")}
Question: {faq.get('question', "Unknown")}
Answer: {faq.get('answer', "Unknown")}"""
        
        return SearchResult(
            source=faq.get("url", ""),
            content=snippet
        )
    
    def parse_news(self, news_item: dict) -> Optional[SearchResult]:
        """
        https://api.search.brave.com/app/documentation/responses#News
        """
        article_description: str = news_item.get("description", "")

        # Throw out items where the description is tiny or doesn't exist.
        if len(article_description) < 5:
            return None

        snippet = f"""News Article Title: {news_item.get('title', "Unknown")}
News Article Description: {article_description}
News Article Age: {news_item.get("age", "Unknown")}
News Article Source: {news_item.get("meta_url", {}).get('hostname', "Unknown")}"""
        
        return SearchResult(
            source=news_item.get("url", ""),
            content=snippet
        )

    @staticmethod
    def remove_strong(web_description: str):
        # this is for cleaning up the brave web descriptions
        return (
            web_description.replace("<strong>", "")
            .replace("</strong>", "")
            .replace("&#x27;", "'")
        )
    
    async def parse_web(self, web_item: dict, query: str) -> SearchResult:
        """
        https://api.search.brave.com/app/documentation/responses#Search
        """
        url = web_item.get("url", "")
        title = web_item.get("title", "")
        description = self.remove_strong(web_item.get("description", ""))
        snippet = f"""Web Page Title: {title}
Web Page URL: {url}
Web Page Description: {description}"""

        try:
            # Currently there is no retry logic here, so if the scrape fails, we just skip it and return the snippet.
            if self.summarize_with_claude:
                content = await scrape_url(url, summarize_with_claude=True,
                                           anthropic_api_key=self.anthropic_api_key, query=query)
            else:
                content = await scrape_url(url)

            if content.startswith('<summary>'):
                snippet+="\nWeb Page Summary: "+content
            else:
                snippet+="\nWeb Page Content: "+content
        except:
            logger.warning(f"Failed to scrape {url}")
        return SearchResult(
            source=url,
            content=snippet
        )


    def raw_search(self, query: str, n_search_results_to_use: int) -> list[SearchResult]:
        """
        Run a search using the BraveAPI and return search results. Here are some details on the Brave API:

        Each search call to the Brave API returns the following fields:
         - faq: Frequently asked questions that are relevant to the search query (only on paid Brave tier).
         - news: News results relevant to the query.
         - web: Web search results relevant to the query.
         - [Thrown Out] videos: Videos relevant to the query.
         - [Thrown Out] locations: Places of interest (POIs) relevant to location sensitive queries.
         - [Thrown Out] infobox: Aggregated information on an entity showable as an infobox.
         - [Thrown Out] discussions: Discussions clusters aggregated from forum posts that are relevant to the query.

        There is also a `mixed` key, which tells us the ranking of the search results.

        We may throw some of these back in, in the future. But we're just going to document the behavior here for now.
        """
        
        # Run the search

        search_response = self.api.search(query)

        # Order everything properly

        correct_ordering = search_response.get("mixed", {}).get("main", [])

        # Extract the results

        faq_items = search_response.get("faq", {}).get("results", [])
        news_items = search_response.get("news", {}).get("results", [])
        web_items = search_response.get("web", {}).get("results", [])

        # Get the search results

        search_results: list[SearchResult] = []
        async_web_parser_loop = asyncio.get_event_loop()
        web_parsing_tasks = [] # We'll queue up the web parsing tasks here, since they're costly

        for item in correct_ordering:
            item_type = item.get("type")
            if item_type == "web":
                web_item = web_items.pop(0)
                ## We'll add a placeholder search result here, and then replace it with the parsed web result later
                url = web_item.get("url", "")
                placeholder_search_result = SearchResult(
                    source=url,
                    content=f"Web Page Title: {web_item.get('title', '')}\nWeb Page URL: {url}\nWeb Page Description: {self.remove_strong(web_item.get('description', ''))}"
                )
                search_results.append(placeholder_search_result)
                ## Queue up the web parsing task
                task = async_web_parser_loop.create_task(self.parse_web(web_item, query))
                web_parsing_tasks.append(task)
            elif item_type == "news":
                parsed_news = self.parse_news(news_items.pop(0))
                if parsed_news is not None:
                    search_results.append(parsed_news)
            elif item_type == "faq":
                parsed_faq = self.parse_faq(faq_items.pop(0))
                search_results.append(parsed_faq)
            if len(search_results) >= n_search_results_to_use:
                break

        ## Replace the placeholder search results with the parsed web results
        web_results = async_web_parser_loop.run_until_complete(asyncio.gather(*web_parsing_tasks))
        web_results_urls = [web_result.source for web_result in web_results]
        for i, search_result in enumerate(search_results):
            url = search_result.source
            if url in web_results_urls:
                search_results[i] = web_results[web_results_urls.index(url)]

        return search_results
    
    def process_raw_search_results(self, results: list[SearchResult]) -> list[list[str]]:
        # We don't need to do any processing here, since we already formatted the results in the `parse` functions.
        processed_search_results = [[result.source, result.content.strip()] for result in results]
        return processed_search_results