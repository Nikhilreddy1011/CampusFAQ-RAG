import logging
from typing import List
import numpy as np
from fastembed import TextEmbedding
from app.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger("campus_faq.embeddings")

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            logger.info("Initializing FastEmbed TextEmbedding with model: %s", EMBEDDING_MODEL_NAME)
            self._model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized float32 embeddings for a list of text strings.
        Shape: (len(texts), EMBEDDING_DIM).
        Normalized so cosine similarity = dot product.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self._get_model()
        raw_embs = list(model.embed(texts))
        embs = np.array(raw_embs, dtype=np.float32)

        # L2 normalize each embedding vector
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normalized_embs = embs / norms
        return normalized_embs

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate a single normalized float32 vector for a student's search query.
        Shape: (EMBEDDING_DIM,).
        """
        model = self._get_model()
        raw_emb = list(model.query_embed(query))[0]
        emb = np.array(raw_emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @staticmethod
    def compute_cosine_similarity(query_emb: np.ndarray, chunk_embs: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity between a single normalized query vector
        and a matrix of normalized chunk vectors.
        """
        if chunk_embs.ndim == 1:
            chunk_embs = chunk_embs.reshape(1, -1)
        sims = np.dot(chunk_embs, query_emb)
        return np.clip(sims, -1.0, 1.0)


# Global singleton instance
embedding_service = EmbeddingService()
