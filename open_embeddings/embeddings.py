"""Core embedding generation functionality."""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates normalized embedding vectors from text using sentence-transformers.

    Optimized for batch processing and HTTP API usage.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model, loading it if necessary."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_single(self, text: str) -> Optional[np.ndarray]:
        """
        Generate a normalized embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector or None if generation fails
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty or whitespace-only text provided")
                return None

            # Generate embedding
            embedding = self.model.encode(text.strip(), convert_to_numpy=True)

            # Normalize to unit length
            normalized = self._normalize_embedding(embedding)

            return normalized

        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate normalized embeddings for a batch of texts.

        More efficient than calling embed_single multiple times.

        Args:
            texts: List of input texts to embed

        Returns:
            List of normalized embedding vectors (None for failed texts)
        """
        if not texts:
            return []

        try:
            # Filter out empty texts but keep track of indices
            valid_texts = []
            text_indices = []

            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    text_indices.append(i)

            if not valid_texts:
                logger.warning("No valid texts provided in batch")
                return [None] * len(texts)

            # Generate embeddings for valid texts
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)

            # Normalize all embeddings
            normalized_embeddings = [
                self._normalize_embedding(emb) for emb in embeddings
            ]

            # Create result list with correct positioning
            results = [None] * len(texts)
            for i, embedding in enumerate(normalized_embeddings):
                original_index = text_indices[i]
                results[original_index] = embedding

            return results

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length.

        Args:
            embedding: Raw embedding vector

        Returns:
            Normalized embedding vector with L2 norm = 1.0
        """
        norm = np.linalg.norm(embedding)

        # Handle zero-magnitude vectors
        if norm == 0:
            logger.warning("Encountered zero-magnitude embedding vector")
            return embedding

        return embedding / norm


# Global model registry for caching loaded models
_model_cache = {}


def get_embedding_generator(model_name: str) -> EmbeddingGenerator:
    """
    Get or create an embedding generator for the specified model.

    Uses caching to avoid reloading models.

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        EmbeddingGenerator instance
    """
    if model_name not in _model_cache:
        _model_cache[model_name] = EmbeddingGenerator(model_name)

    return _model_cache[model_name]


def get_available_models() -> List[str]:
    """
    Get list of commonly available sentence-transformers models.

    Includes popular model families:
    - MiniLM: Fast and efficient models
    - BGE: High-quality BAAI General Embedding models
    - GTE: General Text Embedding models
    - Qwen: Alibaba's high-performance embedding models
    - MPNet: Microsoft's high-performance model

    Returns:
        List of model names
    """
    return [
        # MiniLM family - fast and efficient
        "all-MiniLM-L6-v2",           # 384 dim, good balance
        "all-MiniLM-L12-v2",          # 384 dim, better quality
        "paraphrase-MiniLM-L6-v2",    # 384 dim, good for paraphrases
        "multi-qa-MiniLM-L6-cos-v1",  # 384 dim, good for Q&A

        # BGE family - high quality BAAI models
        "BAAI/bge-small-en-v1.5",     # 384 dim, excellent small model
        "BAAI/bge-base-en-v1.5",      # 768 dim, strong performance
        "BAAI/bge-large-en-v1.5",     # 1024 dim, top performance
        "BAAI/bge-m3",                 # 1024 dim, multilingual

        # GTE family - general text embedding models
        "thenlper/gte-small",          # 384 dim, efficient
        "thenlper/gte-base",           # 768 dim, balanced
        "thenlper/gte-large",          # 1024 dim, high quality

        # Qwen family - Alibaba's embedding models
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # 1536 dim, instruction-tuned
        "Alibaba-NLP/gte-Qwen2-7B-instruct",    # 3584 dim, large model
        "jinaai/jina-embeddings-v2-base-en",     # 768 dim, Jina AI (good alternative)

        # MPNet family - Microsoft's model
        "all-mpnet-base-v2",           # 768 dim, high quality
    ]