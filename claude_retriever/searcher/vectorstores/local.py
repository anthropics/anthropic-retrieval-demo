import os
from claude_retriever.searcher.types import Embedding, HybridEmbedding, SparseEmbeddingData, SearchResult, VectorStore
from scipy.sparse import csr_matrix
import numpy as np
import json


import logging
logger = logging.getLogger(__name__)

############################################
# Local VectorStore implementations
############################################

class LocalVectorStore(VectorStore):
    '''
    Local vectorstores maintain a single embedding matrix.

    How it works:
    - On init, the datastore is loaded from disk if it exists. If it doesn't exist, then an empty datastore is created.
    - When upserting embeddings, the embeddings are written to disk and the datastore is reloaded into memeory.
    -- On disk, the embeddings are stored in JSONL format, where each line has the embedding and text data for an embedding.
    -- In memory, the embeddings are stored as a numpy array.
    - When querying, the query embedding is compared to all embeddings in the datastore using cosine similarity.

    Note that the vectorstore does not contain any logic for creating embeddings. It is assumed that the embeddings are created elsewhere
    using Embedders and passed to the vectorstore for storage and retrieval. The utils.embed_and_upload() is a wrapper to help do this.
    '''
    def __init__(self, disk_path: str):
        if not disk_path.endswith(".jsonl"):
            raise ValueError("disk_path must be .jsonl")
        self.disk_path = disk_path
        self.embeddings = []
        # if there is no file at disk_path, then create an empty file
        if not os.path.exists(disk_path):
            logger.info(f"Creating empty datastore at {disk_path}")
            open(disk_path, "w").close()
        logger.info(f"Loading datastore from {disk_path}")
        self._load_data(disk_path) # If the file is empty, then will load empty embeddings

    def _create_normed_embedding_matrix(self):
        self.embedding_matrix = np.array([emb.embedding for emb in self.embeddings])
        if len(self.embedding_matrix) > 0:
            self.embedding_norms = np.linalg.norm(self.embedding_matrix, axis=1)
        else:
            self.embedding_norms = np.array([])

    def _load_data(self, disk_path: str):
        file_type = disk_path.split(".")[-1]
        if file_type == "jsonl":
            with open(disk_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    text = data["text"]
                    embedding = data["embedding"]
                    self.embeddings.append(Embedding(embedding, text))
        else:
            raise ValueError("Invalid file_type. Supported types: 'jsonl'")
        self._create_normed_embedding_matrix()

    def _nearest_neighbors(self, query_embedding: Embedding, n_search_results_to_use: int) -> list[Embedding]:
        query_vector = query_embedding.embedding
        query_vector = np.array(query_vector)
        query_vector_norm = np.linalg.norm(query_vector)
        cosine_similarities = np.dot(self.embedding_matrix, query_vector) / (self.embedding_norms * query_vector_norm)
        sorted_indices = np.argsort(-cosine_similarities)
        nearest_neighbors = [self.embeddings[i] for i in sorted_indices[:min(n_search_results_to_use, len(self.embeddings))]]
        return nearest_neighbors

    def query(self, query_embedding: Embedding, n_search_results_to_use: int) -> list[SearchResult]:
        if len(self.embeddings) == 0:
            logger.warning("No embeddings in datastore. Returning empty list.")
            return []
        best_matches = self._nearest_neighbors(query_embedding, n_search_results_to_use)
        results = [SearchResult(content=match.text) for match in best_matches]
        return results
    
    def upsert(self, embeddings: list[Embedding]) -> None:
        with open(self.disk_path, "a") as file:
            for item in embeddings:
                output_data = {"embedding": item.embedding, "text": item.text}
                json_str = json.dumps(output_data)
                file.write(json_str + "\n")
        self._load_data(self.disk_path) # Once the embeddings have been written to disk, reload the datastore

class LocalHybridVectorStore(VectorStore):

    '''
    Hybrid vectorstores maintain two embedding matrices: one for dense embeddings and one for sparse embeddings.

    How it works:
    - On init, the datastore is loaded from disk if it exists. If it doesn't exist, then an empty datastore is created.
    - When upserting embeddings, the embeddings are written to disk and the datastore is reloaded into memeory.
    -- On disk, the embeddings are stored in JSONL format, where each line has the dense, sparse and text data for an embedding.
    -- In memory, the embeddings are stored as two separate matrices: one for dense embeddings and one for sparse embeddings. The latter is stored as a scipy.sparse.csr_matrix.
    - When querying, the query embedding is compared to all dense embeddings and all sparse embeddings in the datastore. They are combined using either:
    -- split_return: half of the expected number of results are returned from the top dense matches and half from the top sparse matches (after deduplication).
    -- weighted: the top n_search_results_to_use results are returned from the weighted combination of dense and sparse cosine similarities.

    Note that the vectorstore does not contain any logic for creating embeddings. It is assumed that the embeddings are created elsewhere 
    using Embedders and passed to the vectorstore for storage and retrieval. The utils.embed_and_upload() is a wrapper to help do this.
    '''

    def __init__(self, disk_path: str):
        if not disk_path.endswith(".jsonl"):
            raise ValueError("disk_path must be .jsonl")
        self.disk_path = disk_path
        self.embeddings = []
        # if there is no file at disk_path, then create an empty file
        if not os.path.exists(disk_path):
            logger.info(f"Creating empty datastore at {disk_path}")
            open(disk_path, "w").close()
        logger.info(f"Loading datastore from {disk_path}")
        self._load_data(disk_path)

    @staticmethod
    def sparse_embedding_data_to_sparse_matrix(sparse_embedding_index: list[SparseEmbeddingData]) -> csr_matrix:
        """
        Converts a list of SparseEmbeddingData to a sparse matrix.

        The sparse matrix format is a scipy.sparse.csr_matrix, which is a sparse matrix stored in Compressed Sparse Row format.
        """
        if not sparse_embedding_index:
            return csr_matrix((0, 0))

        # Check that all SparseEmbeddingData objects have the same max_index
        max_index = sparse_embedding_index[0].max_index
        for sparse_embedding in sparse_embedding_index:
            if sparse_embedding.max_index != max_index:
                raise ValueError("SparseEmbeddingData objects must have the same max_index")

        # Create sparse matrix
        rows, cols, data = [], [], []
        for idx, sparse_embedding in enumerate(sparse_embedding_index):
            rows.extend([idx] * len(sparse_embedding.indices))
            cols.extend(sparse_embedding.indices)
            data.extend(sparse_embedding.values)

        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(sparse_embedding_index), max_index + 1))
        return sparse_matrix

    def _create_normed_embedding_matrices(self):

        '''
        Hybrid vectorstores maintain two embedding matrices: one for dense embeddings and one for sparse embeddings.
        '''

        self.dense_embedding_matrix = np.array([emb.embedding for emb in self.embeddings])
        if len(self.dense_embedding_matrix) > 0:
            self.dense_embedding_norms = np.linalg.norm(self.dense_embedding_matrix, axis=1)
        else:
            self.dense_embedding_norms = np.array([])

        self.sparse_embedding_matrix = self.sparse_embedding_data_to_sparse_matrix([emb.sparse_embedding for emb in self.embeddings])
        if self.sparse_embedding_matrix.shape[0] > 0:
            self.sparse_embedding_norms = np.linalg.norm(self.sparse_embedding_matrix.toarray(), axis=1)
        else:
            self.sparse_embedding_norms = np.array([])


    def _load_data(self, disk_path: str):
        file_type = disk_path.split(".")[-1]
        if file_type == "jsonl":
            with open(disk_path, "r") as f:
                for (i, line) in enumerate(f):
                    data = json.loads(line)
                    if "sparse_embedding" not in data.keys():
                        raise ValueError(f"Invalid jsonl file. 'sparse_embedding' key is missing on line {i}.")
                    text = data["text"]
                    dense_embedding = data["embedding"]
                    sparse_embedding_idx = data["sparse_embedding"]
                    expected_keys = {"indices", "values", "max_index"}
                    if not expected_keys.issubset(set(sparse_embedding_idx.keys())):
                        raise ValueError(f"Invalid jsonl file. Some of expected keys {expected_keys} are missing from sparse_embedding on line {i}.")
                    sparse_embedding = SparseEmbeddingData(sparse_embedding_idx["indices"],
                                                           sparse_embedding_idx["values"],
                                                           sparse_embedding_idx["max_index"])
                    self.embeddings.append(HybridEmbedding(embedding=dense_embedding, sparse_embedding=sparse_embedding, text=text))
        else:
            raise ValueError("Invalid file_type. Supported types: 'jsonl'")
        self._create_normed_embedding_matrices()

    def _get_dense_cosine_similarities(self, query_embedding: HybridEmbedding) -> np.ndarray:

        '''
        This method returns the cosine similarities between the query embedding and all dense embeddings in the datastore.
        '''

        query_dense_vector = query_embedding.embedding
        query_dense_vector = np.array(query_dense_vector)
        query_dense_vector_norm = np.linalg.norm(query_dense_vector)
        dense_inner_prod = np.dot(self.dense_embedding_matrix, query_dense_vector)
        dense_cosine_similarities = dense_inner_prod / (self.dense_embedding_norms * query_dense_vector_norm)
        return dense_cosine_similarities
    
    def _get_sparse_cosine_similarities(self, query_embedding: HybridEmbedding) -> np.ndarray:

        '''
        This method returns the cosine similarities between the query embedding and all sparse embeddings in the datastore.
        '''

        query_sparse_vector = self.sparse_embedding_data_to_sparse_matrix([query_embedding.sparse_embedding])
        query_sparse_vector_norm = np.linalg.norm(query_sparse_vector.toarray())
        sparse_inner_prod = np.dot(self.sparse_embedding_matrix, query_sparse_vector.T).T.toarray().flatten()
        sparse_cosine_similarities = sparse_inner_prod / (self.sparse_embedding_norms * query_sparse_vector_norm)
        return sparse_cosine_similarities

    def _query_split_return(self, query_embedding: HybridEmbedding, n_search_results_to_use: int) -> list[HybridEmbedding]:

        '''
        This method returns half of the expected number of results from the top dense matches and half from the top sparse matches.
        '''

        n_dense_results = int(n_search_results_to_use / 2)
        n_sparse_results = n_search_results_to_use - n_dense_results

        dense_cosine_results = self._get_dense_cosine_similarities(query_embedding)
        sparse_cosine_results = self._get_sparse_cosine_similarities(query_embedding)

        dense_sorted_indices = np.argsort(-dense_cosine_results)
        sparse_sorted_indices = np.argsort(-sparse_cosine_results)
        
        sorted_indices = dense_sorted_indices[:min(n_dense_results, len(self.embeddings))]
        sparse_sorted_indices = [i for i in sparse_sorted_indices if i not in sorted_indices] # remove the already chosen dense results
        sorted_indices = np.concatenate((sorted_indices, sparse_sorted_indices[:min(n_sparse_results, len(sparse_sorted_indices))]))

        nearest_neighbors = [self.embeddings[i] for i in sorted_indices]
        return nearest_neighbors
    
    def _query_weighted(self, query_embedding: HybridEmbedding, n_search_results_to_use: int, sparse_weight: float = 0.5) -> list[HybridEmbedding]:

        '''
        This method returns the top n_search_results_to_use results from the weighted combination of dense and sparse cosine similarities.
        '''

        if sparse_weight < 0.0 or sparse_weight > 1.0:
            raise ValueError("sparse_weight must be between 0.0 and 1.0")
        
        dense_cosine_results = self._get_dense_cosine_similarities(query_embedding)
        sparse_cosine_results = self._get_sparse_cosine_similarities(query_embedding)

        cosine_similarities = (1-sparse_weight) * dense_cosine_results + sparse_weight * sparse_cosine_results

        sorted_indices = np.argsort(-cosine_similarities)
        nearest_neighbors = [self.embeddings[i] for i in sorted_indices[:min(n_search_results_to_use, len(self.embeddings))]]
        return nearest_neighbors


    def query(self, query_embedding: HybridEmbedding, n_search_results_to_use: int, query_strategy: str = "split_return", sparse_weight: float = 0.5) -> list[SearchResult]:
        if len(self.embeddings) == 0:
            logger.warning("No embeddings in datastore. Returning empty list.")
            return []
        if query_strategy == "split_return":
            best_matches = self._query_split_return(query_embedding, n_search_results_to_use)
        elif query_strategy == "weighted":
            best_matches = self._query_weighted(query_embedding, n_search_results_to_use, sparse_weight=sparse_weight)
        else:
            best_matches = self._query_weighted(query_embedding, n_search_results_to_use, sparse_weight=0.0) # default to dense only
        results = [SearchResult(content=match.text) for match in best_matches]
        return results
    
    def upsert(self, embeddings: list[HybridEmbedding]) -> None:
        with open(self.disk_path, "a") as file:
            for item in embeddings:
                sparse_embedding_idx = {"indices": item.sparse_embedding.indices, "values": item.sparse_embedding.values, "max_index": item.sparse_embedding.max_index}
                output_data = {"embedding": item.embedding, "sparse_embedding": sparse_embedding_idx, "text": item.text}
                json_str = json.dumps(output_data)
                file.write(json_str + "\n")
        self._load_data(self.disk_path) # Once the embeddings have been written to disk, reload the datastore
