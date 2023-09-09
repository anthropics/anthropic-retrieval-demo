from claude_retriever.searcher.types import Embedding, HybridEmbedding, SearchResult, VectorStore
import pinecone
from more_itertools import chunked

import logging
logger = logging.getLogger(__name__)


############################################
# Pinecone VectorStore implementations
############################################

class PineconeVectorStore(VectorStore):
    '''
    Pinecone vectorstores maintain a single embedding matrix.
    
    How it works:
    - On init, the Pinecone index is loaded (this assumes that the Pinecone index already exists).
    - When upserting embeddings, the embeddings are upserted into the Pinecone index.
    -- The embeddings are stored as a list of ids, vectors, and metadatas. Metadatas are used to store the text data for each embedding; Pinecone indices do not store text data by default.
    -- The ids are the index of the embedding in the Pinecone index.
    - When querying, the query embedding is compared to all embeddings in the Pinecone index using the similarity specified when the index was created.

    Note that the vectorstore does not contain any logic for creating embeddings. It is assumed that the embeddings are created elsewhere
    using Embedders and passed to the vectorstore for storage and retrieval. The utils.embed_and_upload() is a wrapper to help do this.
    '''
    def __init__(self, api_key: str, environment: str, index: str):
        self.api_key = api_key
        self.environment = environment
        self.index = index
        self.pinecone_index = self._init_pinecone_index()
        self.pinecone_index_dimensions = self.pinecone_index.describe_index_stats().dimension

    def _init_pinecone_index(self):
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment,
        )
        if self.index not in pinecone.list_indexes():
            raise ValueError(f"Pinecone index {self.index} does not exist")
        return pinecone.Index(self.index)

    def query(self, query_embedding: Embedding, n_search_results_to_use: int = 10) -> list[SearchResult]:
        if len(query_embedding.embedding) != self.pinecone_index_dimensions:
            raise ValueError(f"Query embedding dimension {len(query_embedding.embedding)} does not match Pinecone index dimension {self.pinecone_index_dimensions}")
        results = self.pinecone_index.query(
            vector=query_embedding.embedding, top_k=n_search_results_to_use, include_metadata=True
        )
        results=[SearchResult(content=match['metadata']['text']) for match in results.matches]
        return results

    def upsert(self, embeddings: list[Embedding], upsert_batch_size: int = 128) -> None:
        '''
        This method upserts embeddings into the Pinecone index in batches of size upsert_batch_size.

        Since Pinecone indices uniquely identify embeddings by their ids,
        we need to keep track of the current index size and update the id counter correspondingly.
        '''
        embedding_chunks = chunked(embeddings, n=upsert_batch_size) # split embeddings into chunks of size upsert_batch_size
        current_index_size = self.pinecone_index.describe_index_stats()['total_vector_count'] # get the current index size from Pinecone
        i = 0 # keep track of the current index in the current batch
        for emb_chunk in embedding_chunks:
            # for each chunk of size upsert_batch_size, create a list of ids, vectors, and metadatas, and upsert them into the Pinecone index
            ids = [str(current_index_size+1+i) for i in range(i,i+len(emb_chunk))]
            vectors = [emb.embedding for emb in emb_chunk]
            metadatas = [{'text': emb.text} for emb in emb_chunk]
            records = list(zip(ids, vectors, metadatas))
            self.pinecone_index.upsert(vectors=records)
            i += len(emb_chunk) 

class PineconeHybridVectorStore(PineconeVectorStore):

    '''
    Pinecone hybrid vectorstores maintain two embedding matrices: one for dense embeddings and one for sparse embeddings.
    
    How it works:
    - On init, the Pinecone index is loaded (this assumes that the Pinecone index already exists).
    - When upserting embeddings, the embeddings are upserted into the Pinecone index.
    -- The embeddings are stored as a list of ids, vectors, and metadatas. Metadatas are used to store the text data for each embedding; Pinecone indices do not store text data by default.
    -- The ids are the index of the embedding in the Pinecone index.
    -- The sparse embeddings are stored as a list of sparse vectors, where each sparse vector is a dict of indices and values.
    - When querying, the query embedding is compared to all dense embeddings and all sparse embeddings in the Pinecone index. They are combined using either:
    -- split_return: half of the expected number of results are returned from the top dense matches and half from the top sparse matches (after deduplication).
    -- weighted: the top n_search_results_to_use results are returned from the weighted combination of dense and sparse cosine similarities.

    Note that the vectorstore does not contain any logic for creating embeddings. It is assumed that the embeddings are created elsewhere
    using Embedders and passed to the vectorstore for storage and retrieval. The utils.embed_and_upload() is a wrapper to help do this.
    '''

    def _query_split_return(self, query_embedding: HybridEmbedding, n_search_results_to_use: int) -> list[SearchResult]:

        '''
        This method returns half of the expected number of results from the top dense matches and half from the top sparse matches.
        '''

        dense_results = self.pinecone_index.query(
            vector=query_embedding.embedding, top_k=n_search_results_to_use, include_metadata=True
        )

        sparse_results = self.pinecone_index.query(
            vector=[0.]*len(query_embedding.embedding),
            sparse_vector={
                'indices': query_embedding.sparse_embedding.indices,
                'values': query_embedding.sparse_embedding.values,
            },
            top_k=n_search_results_to_use,
            include_metadata=True
        )

        n_dense_results = int(n_search_results_to_use / 2)
        n_sparse_results = n_search_results_to_use - n_dense_results

        dense_results = dense_results.matches[:min(n_dense_results, len(dense_results.matches))]
        dense_results_texts = [result['metadata']['text'] for result in dense_results]

        # Remove the already chosen dense results
        sparse_results = [sparse_result for sparse_result in sparse_results.matches if sparse_result['metadata']['text'] not in dense_results_texts]
        sparse_results = sparse_results[:min(n_sparse_results, len(sparse_results))]

        results = dense_results + sparse_results
        results = [SearchResult(content=match['metadata']['text']) for match in results]

        return results
    
    def _query_weighted(self, query_embedding: HybridEmbedding, n_search_results_to_use: int, sparse_weight: float = 0.5) -> list[SearchResult]:

        '''
        This method returns the top n_search_results_to_use results from the weighted combination of dense and sparse cosine similarities.
        '''

        def hybrid_score_norm(dense, sparse, sparse_weight: float):
            """Hybrid score using a convex combination

            (1 - sparse_weight) * dense + sparse_weight * sparse

            Args:
                dense: Array of floats representing
                sparse: a dict of `indices` and `values`
                sparse_weight: scale between 0 and 1
            """
            if sparse_weight < 0 or sparse_weight > 1:
                raise ValueError("Alpha must be between 0 and 1")
            hs = {
                'indices': sparse['indices'],
                'values':  [v * (1 - sparse_weight) for v in sparse['values']]
            }
            return [v * sparse_weight for v in dense], hs

        sparse_query_vector = {
            'indices': query_embedding.sparse_embedding.indices,
            'values': query_embedding.sparse_embedding.values
        }

        dense_query_vector = query_embedding.embedding

        hdense, hsparse = hybrid_score_norm(dense_query_vector, sparse_query_vector, sparse_weight)        

        results = self.pinecone_index.query(
            vector=hdense,
            sparse_vector=hsparse,
            top_k=n_search_results_to_use,
            include_metadata=True
        )

        results = [SearchResult(content=match['metadata']['text']) for match in results.matches]

        return results
        
    def query(self, query_embedding: HybridEmbedding, n_search_results_to_use: int = 10, query_strategy: str = "split_return", sparse_weight: float = 0.5) -> list[SearchResult]:
        if len(query_embedding.embedding) != self.pinecone_index_dimensions:
            raise ValueError(f"Query embedding dimension {len(query_embedding.embedding)} does not match Pinecone index dimension {self.pinecone_index_dimensions}")
        if query_strategy == "split_return":
            results = self._query_split_return(query_embedding, n_search_results_to_use)
        elif query_strategy == "weighted":
            results = self._query_weighted(query_embedding, n_search_results_to_use, sparse_weight=sparse_weight)
        else:
            results = self._query_weighted(query_embedding, n_search_results_to_use, sparse_weight=0.0) # default to dense only
        return results

    def upsert(self, embeddings: list[HybridEmbedding], upsert_batch_size: int = 128) -> None:
        '''
        This method upserts embeddings into the Pinecone index in batches of size upsert_batch_size.

        Since Pinecone indices uniquely identify embeddings by their ids,
        we need to keep track of the current index size and update the id counter correspondingly.
        '''
        embedding_chunks = chunked(embeddings, n=upsert_batch_size) # split embeddings into chunks of size upsert_batch_size
        current_index_size = self.pinecone_index.describe_index_stats()['total_vector_count'] # get the current index size from Pinecone
        i = 0 # keep track of the current index in the current batch
        for emb_chunk in embedding_chunks:
            # for each chunk of size upsert_batch_size, create a list of ids, vectors, and metadatas, and upsert them into the Pinecone index
            ids = [str(current_index_size + 1 + i) for i in range(i,i+len(emb_chunk))]
            vectors = [emb.embedding for emb in emb_chunk]
            metadatas = [{'text': emb.text} for emb in emb_chunk]
            sparse_values = [{'indices': emb.sparse_embedding.indices, 'values': emb.sparse_embedding.values} for emb in emb_chunk]
            records = [{'id': i,
                        'values': v,
                        'metadata': m,
                        'sparse_values': s} for i,v,m,s
                        in zip(ids, vectors, metadatas, sparse_values)]
            self.pinecone_index.upsert(vectors=records)
            i += len(emb_chunk)