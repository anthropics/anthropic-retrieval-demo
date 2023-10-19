from claude_retriever.searcher.types import Embedding, HybridEmbedding, SearchResult, VectorStore, Embedder
from claude_retriever.searcher.embedders.local import LocalEmbedder, LocalHybridEmbedder
from constants import DEFAULT_EMBEDDER, DEFAULT_SPARSE_EMBEDDER
from typing import Optional
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

from pymongo import MongoClient

############################################
# MongoDB Atlas VectorStore implementations
############################################

class MongoDBAtlasVectorStore(VectorStore):
    '''
    MongDB Atlas vectorstores maintains Vector Indexes alongside Document Data schema stored in the ODS(Operational Data Store).
    
    How it works:
    - On init, the MongoDB Atlas Connections are created and a collection is created/initalized.
    - Use the UI to create vector search indexes if not already created(https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
    - When upserting embeddings, the embeddings are upserted into the initialized mongodb collection.
    -- The embeddings are stored as a field level entity alongside the other meta information. Metadatas stored are used to map the text data for each respective embedding.
    - When querying, the query embedding is compared to all embeddings in the Vector search index using the similarity specified when the vector search index was created.

    Note that the vectorstore does not contain any logic for creating embeddings. It is assumed that the embeddings are created elsewhere
    using Embedders and passed to the vectorstore for storage and retrieval. The class has the method self_load_index_embeddings is a wrapper to help do this.
    '''

    def __init__(self, conn_str, db_name, col_name, embedding: Optional[Embedder]= LocalEmbedder(DEFAULT_EMBEDDER) , index_name='default', text_key='text', embedding_key='embedding') -> None:
        """
        Args:
            conn_str: MongoDB connection string to init connection client
            db_name: MongoDB database name
            col_name: MongoDB collection name
            embedding: Text embedding model to use.
            text_key: MongoDB field that will contain the text for each
                document.
            embedding_key: MongoDB field that will contain the embedding for
                each document.
            index_name: Name of the Atlas Search index.
        """
        super().__init__()
        self._client = MongoClient(conn_str)
        self._collection = self._client[db_name][col_name]
        self.embedding = embedding
        self._index_name = index_name
        self.text_key = text_key
        self.embedding_key = embedding_key

        self._init_collection()

    def _init_collection(self):
        if not self._collection.find_one():
            self._collection.insert_one({"text":'', "embedding":[0]*self.embedding.dim})
            self._collection.delete_many({})

    def chunker(self, ip, size):
        return (ip[pos:pos + size] for pos in range(0, len(ip), size))

    def _load_index_embeddings(self,documents, batch_size):
        if not self._collection.find_one():
            if len(documents)>0:
                for docs in tqdm(self.chunker(documents, batch_size)):
                    texts = [doc[self.text_key] for doc in docs if self.text_key in doc]
                    idocs = []
                    for embedding in self.embedding.embed_batch(texts):
                        idocs += [{self.text_key: embedding.text, self.embedding_key:embedding.embedding}]
                    self._collection.insert_many(idocs)

    def _get_vector_search_query(self,query_vector, k=10):
        ## TODO update the to $vectorsearch syntax when the feature is GA.
        return [{
            "$search": {
                "index": "default",
                "knnBeta": {
                "vector": query_vector.embedding,
                "path": self.embedding_key,
                "k": k
                }
            }
            },
            {"$project":{
                "score":{
                            '$meta': 'searchScore'
                        },
                self.embedding_key:0,
                "_id": 0
            }}]
    

    def query(self, query_embedding: Embedding, n_search_results_to_use: int = 10) -> list[SearchResult]:
        """
        Runs a query using the vector store and returns the results.

        :param query: String input to query the data with.
        :param n_search_results_to_use: The number of results to return.
        """
        if not query_embedding.embedding:
            query_embedding.embedding = self.emmbedding.embed(query_embedding.text)     
        mdb_query = self._get_vector_search_query(query_embedding,n_search_results_to_use)
        best_matches = list(self._collection.aggregate(mdb_query))
        return [SearchResult(content=match["text"]) for match in best_matches]
    
    def upsert(self, embeddings: list[Embedding]) -> None:
        for item in embeddings:
            if not item.embedding:
                item.embedding = self.emmbedding.embed(item.text)
            self._collection.upsert({self.text_key:item.text}, {"$set":{self.embedding_key:item.embedding}})
