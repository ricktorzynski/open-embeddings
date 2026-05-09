"""Tests for the embedding generation functionality."""

import numpy as np
import pytest
from open_embeddings.embeddings import EmbeddingGenerator, get_embedding_generator


@pytest.fixture
def generator():
    """Create an embedding generator for testing."""
    return EmbeddingGenerator()


def test_embedding_generator_initialization(generator):
    """Test that embedding generator initializes correctly."""
    assert generator.model_name == "all-MiniLM-L6-v2"
    assert generator.model is not None
    assert generator.dimension > 0


def test_embed_single_returns_normalized_array(generator):
    """Test that embed_single returns a normalized numpy array."""
    text = "This is a test sentence for embedding generation."
    embedding = generator.embed_single(text)

    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0

    # Check normalization (L2 norm should be approximately 1.0)
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-6)


def test_embed_single_handles_empty_text(generator):
    """Test that embed_single handles empty text gracefully."""
    assert generator.embed_single("") is None
    assert generator.embed_single("   ") is None
    assert generator.embed_single(None) is None


def test_embed_batch_consistent_dimensions(generator):
    """Test that batch embeddings have consistent dimensions."""
    texts = [
        "Short text",
        "This is a longer text with more words and content for embedding",
        "Another test sentence"
    ]

    embeddings = generator.embed_batch(texts)

    assert len(embeddings) == len(texts)

    # Filter out None embeddings
    valid_embeddings = [emb for emb in embeddings if emb is not None]

    assert len(valid_embeddings) == 3  # All should succeed

    # Check all have same dimensions
    dimensions = [len(emb) for emb in valid_embeddings]
    assert len(set(dimensions)) == 1  # All dimensions should be equal


def test_embed_batch_normalization(generator):
    """Test that all batch embeddings are normalized."""
    texts = ["Text one", "Text two", "Text three"]
    embeddings = generator.embed_batch(texts)

    for embedding in embeddings:
        if embedding is not None:
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, atol=1e-6)


def test_embed_batch_handles_mixed_valid_invalid(generator):
    """Test batch processing with mix of valid and invalid texts."""
    texts = ["Valid text", "", "Another valid text", "   ", "Final valid text"]
    embeddings = generator.embed_batch(texts)

    assert len(embeddings) == 5
    assert embeddings[0] is not None  # Valid
    assert embeddings[1] is None      # Empty
    assert embeddings[2] is not None  # Valid
    assert embeddings[3] is None      # Whitespace
    assert embeddings[4] is not None  # Valid


def test_embed_batch_empty_list(generator):
    """Test batch processing with empty list."""
    embeddings = generator.embed_batch([])
    assert embeddings == []


def test_get_embedding_generator_caching():
    """Test that get_embedding_generator caches models."""
    gen1 = get_embedding_generator("all-MiniLM-L6-v2")
    gen2 = get_embedding_generator("all-MiniLM-L6-v2")

    # Should return the same instance due to caching
    assert gen1 is gen2


def test_different_models_different_instances():
    """Test that different models create different instances."""
    gen1 = get_embedding_generator("all-MiniLM-L6-v2")
    # Note: This might fail if the second model isn't available
    # In a real test environment, you'd mock this or ensure models are available
    try:
        gen2 = get_embedding_generator("all-MiniLM-L12-v2")
        assert gen1 is not gen2
        assert gen1.model_name != gen2.model_name
    except Exception:
        # Skip if model not available
        pytest.skip("Second model not available for testing")


def test_embedding_consistency(generator):
    """Test that the same text produces consistent embeddings."""
    text = "This is a consistency test"

    embedding1 = generator.embed_single(text)
    embedding2 = generator.embed_single(text)

    assert embedding1 is not None
    assert embedding2 is not None

    # Embeddings should be very similar (allowing for floating point precision)
    cosine_sim = np.dot(embedding1, embedding2)
    assert cosine_sim > 0.99  # Should be very close to 1.0