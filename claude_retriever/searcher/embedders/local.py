from claude_retriever.searcher.types import (
    Embedding,
    SparseEmbeddingData,
    HybridEmbedding,
    Embedder,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import torch
from typing import cast


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = SentenceTransformer(model_name, device=self.device)

        dim = self.model.get_sentence_embedding_dimension()
        self.dim = cast(int, dim)

    def embed(self, text: str) -> Embedding:
        emb = self.embed_batch([text])
        return emb[0]

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        assert isinstance(embeddings, np.ndarray)
        embeddings = [embedding.tolist() for embedding in embeddings]
        return [
            Embedding(embedding=embedding, text=text)
            for embedding, text in zip(embeddings, texts)
        ]


class LocalHybridEmbedder(Embedder):
    def __init__(self, dense_model_name: str, sparse_model_name: str):
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.dense_model = SentenceTransformer(dense_model_name, device=self.device)
        dense_dim = self.dense_model.get_sentence_embedding_dimension()
        self.dense_dim = cast(int, dense_dim)

        self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(self.sparse_model_name)
        self.sparse_model.to(self.device)
        self.sparse_dim = self.sparse_model.config.vocab_size

    def _sparse_encode(self, texts: list[str]) -> list[SparseEmbeddingData]:
        with torch.no_grad():
            tokens = self.sparse_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            vec = torch.max(
                torch.log(1 + torch.relu(self.sparse_model(**tokens).logits))
                * tokens.attention_mask.unsqueeze(-1),
                dim=1,
            ).values.cpu()
            indices = [v.nonzero().flatten() for v in vec]
            weights = [v[idx] for v, idx in zip(vec, indices)]
            sparse_embeddings = [
                SparseEmbeddingData(
                    indices=c.tolist(), values=w.tolist(), max_index=self.sparse_dim
                )
                for c, w in zip(indices, weights)
            ]
        return sparse_embeddings

    def embed(self, text: str) -> HybridEmbedding:
        emb = self.embed_batch([text])
        return emb[0]

    def embed_batch(self, texts: list[str]) -> list[HybridEmbedding]:
        # Get dense embeddings
        dense_embeddings = self.dense_model.encode(texts, show_progress_bar=False)
        assert isinstance(dense_embeddings, np.ndarray)
        dense_embeddings = [embedding.tolist() for embedding in dense_embeddings]
        # Get sparse embeddings
        sparse_embeddings = self._sparse_encode(texts)
        assert isinstance(sparse_embeddings[0], SparseEmbeddingData)
        # Combine dense and sparse embeddings
        return [
            HybridEmbedding(
                embedding=embedding, sparse_embedding=sparse_embedding, text=text
            )
            for embedding, sparse_embedding, text in zip(
                dense_embeddings, sparse_embeddings, texts
            )
        ]
