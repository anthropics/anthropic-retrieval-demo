import os
import json
import re
import anthropic
from anthropic import Anthropic, AsyncAnthropic
from typing import Optional
from searcher.types import Embedder, VectorStore, Tool, SearchTool, SearchResult
from searcher.embedders.local import LocalEmbedder
from constants import DEFAULT_EMBEDDER
from tqdm import tqdm
from dataclasses import dataclass
import aiohttp

import logging
logger = logging.getLogger(__name__)

# Formatting search results
def format_results(extracted: list[list[str]]) -> str:
        """
        Joins and formats the extracted search results as a string.

        :param extracted: The extracted search results to format.
        """
        result = "\n".join(
            [
                f'<item index="{i+1}">\n<source>{r[0]}</source>\n<page_content>\n{r[1]}\n</page_content>\n</item>'
                for i, r in enumerate(extracted)
            ]
        )
        return result

def format_results_full(extracted: list[list[str]]) -> str:
    """
    Formats the extracted search results as a string, including the <search_results> tags.

    :param extracted: The extracted search results to format.
    """
    return f"\n<search_results>\n{format_results(extracted)}\n</search_results>"


# Chunking and uploading

@dataclass
class Document:
    """
    A single document.
    """
    text: str
    metadata: Optional[dict] = None

## Embedding and uploading

def embed_and_upload(
        input_file: str,
        vectorstore: VectorStore,
        embedder: Optional[Embedder] = None,
        tokens_per_chunk: int = 384,
        stride: Optional[int] = None,
        batch_size: int = 128) -> None:
    
    if embedder is None:
        logger.info(f"Using default embedder: {DEFAULT_EMBEDDER}")
        embedder = LocalEmbedder(DEFAULT_EMBEDDER)

    # Load the documents
    documents: list[Document] = []
    file_type = input_file.split(".")[-1]
    if file_type == "jsonl":
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                text = data["text"]
                if text is None:
                    raise ValueError(f"Invalid jsonl file. 'text' key is missing on line {i}")
                metadata = data.get("metadata", None)
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)
    else:
        raise ValueError("Invalid file_type. Supported types: 'jsonl'")
    
    # Chunk the documents
    chunked_documents = []
    for document in documents:
        chunks = chunk_document(document, tokens_per_chunk, stride)
        chunked_documents += chunks

    # Embed and upload the documents
    bar = tqdm(total=len(chunked_documents), desc="Embedding and uploading documents", leave=True)
    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i:i + batch_size]
        batch_embeddings = embedder.embed_batch([doc.text for doc in batch])
        vectorstore.upsert(batch_embeddings)
        bar.update(len(batch))


def chunk_document(document: Document, tokens_per_chunk: int, stride: Optional[int] = None) -> list[Document]:

    if stride is None:
        stride = tokens_per_chunk

    tok = Anthropic().get_tokenizer()

    raw_text = document.text
    tokenized_text = tok.encode(raw_text).ids

    chunks = []
    for i in range(0, len(tokenized_text), stride):
        chunk = tokenized_text[i:i + tokens_per_chunk]
        chunk_text = tok.decode(chunk)
        chunk_document = Document(text=chunk_text, metadata=document.metadata)
        chunks.append(chunk_document)
    return chunks

## Elasticsearch uploading

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def upload_to_elasticsearch(
        input_file: str,
        index_name: str,
        cloud_id: str,
        api_key_id: str,
        api_key: str) -> None:
    
    # Load the documents

    documents: list[Document] = []
    file_type = input_file.split(".")[-1]
    if file_type == "jsonl":
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                text = data["text"]
                if text is None:
                    raise ValueError(f"Invalid jsonl file. 'text' key is missing on line {i}")
                metadata = data.get("metadata", None)
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)
    else:
        raise ValueError("Invalid file_type. Supported types: 'jsonl'")
    
    # Upload the documents

    ## Create the Elasticsearch client

    es = Elasticsearch(
        cloud_id=cloud_id,
        api_key=(api_key_id, api_key),
    )

    ## Upload the documents

    def docs_to_generator():
        for i, document in enumerate(documents):
            yield {
                "_index": index_name,
                "_id": i,
                "text": document.text,
                "metadata": document.metadata
            }
    
    bulk(es, docs_to_generator())
    es.indices.refresh(index=index_name)

# Extract content, potentially using language models

from bs4 import BeautifulSoup

async def scrape_url(url: str, summarize_with_claude: bool = False,
               query: Optional[str] = None,
               anthropic_api_key: Optional[str] = None,
               missing_content_placeholder: str = "CONTENT NOT AVAILABLE") -> str:
    content = await get_url_content(url)
    if content:
        if summarize_with_claude:
            if anthropic_api_key is None:
                try:
                    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
                except KeyError:
                    raise ValueError("anthropic_api_key must be provided if llm_extract is True")
            try:
                content = await claude_extract(content, query, anthropic_api_key)
            except Exception as e:
                logger.warning(f"Failed to extract with Claude. Falling back to raw content. Error: {e}")
    else:
        content = missing_content_placeholder
    return content

async def get_url_content(url: str) -> Optional[str]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text(strip=True, separator='\n')
                return text
    return None

async def claude_extract(content: str, query: Optional[str], anthropic_api_key: str, max_tokens_to_read: int = 20_000) -> str:

    # Get first max_tokens_to_read words tokens of content

    client = Anthropic(api_key=anthropic_api_key)
    tokenizer = client.get_tokenizer()
    tokenized_content = tokenizer.encode(content).ids
    if len(tokenized_content) > max_tokens_to_read:
        logger.info(f"Truncating content from {len(tokenized_content)} tokens to {max_tokens_to_read} tokens")
        content = tokenizer.decode(tokenized_content[:max_tokens_to_read]).strip()

    # Generate prompt

    prompt = f"""{anthropic.HUMAN_PROMPT} Here is the content of a web page:
<content>
{content}
</content>"""

    if query:
        prompt += f"""
Here is a search query a user made which resulted in this page:

<query>
{query}
</query>

<instructions>
* Please provide a summary of the web page that is relevant to the query.
* Please made the summary as concise as possible, in bullet points, and include all the information that might be relevant to the query.
"""

    else:

        prompt += f"""

<instructions>
* Please provide a summary of the web page, as concisely as possible, in bullet points."""

    prompt += f"""
* Please do not introduce any additional information beyond what's in the web page.
* Please do not make any guesses about the contents of the web page. Simply summarize based on the given information content of the web page.
* If the content of the web page is not meaningful, relevant, or understandable then don't write anything.
* IMPORTANT: You are going to simulate the output of a web page. Therefore your response should look indistinguishable from what one might read on a web page. This means you should neither refer to yourself in any way in the response nor make it apparent that you are providing a summary. You should not explicitly mention in any way that you are providing a simulated output of a web page.
* IMPORTANT: Please do not ask for feedback on your summary in any way at the end of your response.
* IMPORTANT: Please do not mention that you are providing a summary, or mention "summary" in any other way.
</instructions>{anthropic.AI_PROMPT} Based on the given content{' and query' if query else ''}, the summary of the page would be:
<summary>"""

    logger.info(f"Triggering a Claude extract for a {len(tokenizer.encode(content).ids)} token document")

    client = AsyncAnthropic(api_key=anthropic_api_key)
    response = await client.completions.create(
        prompt=prompt,
        max_tokens_to_sample=512,
        temperature=0.0,
        model="claude-instant-v1",
        stop_sequences=["</summary>", anthropic.HUMAN_PROMPT]
    )

    # Extract summary

    completion = response.completion
    if not completion.endswith("</summary>"):
        completion += "</summary>"
    return '<summary>'+completion
