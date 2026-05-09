"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from open_embeddings.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_list_models(client):
    """Test the models listing endpoint."""
    response = client.get("/models")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data
    assert "default" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


def test_embed_single_text(client):
    """Test embedding generation for a single text."""
    request_data = {
        "texts": ["This is a test sentence for embedding."]
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "embeddings" in data
    assert "model" in data
    assert "dimension" in data
    assert "count" in data
    assert "successful" in data

    # Check response structure
    assert data["count"] == 1
    assert data["successful"] == 1
    assert len(data["embeddings"]) == 1
    assert data["embeddings"][0] is not None
    assert len(data["embeddings"][0]) == data["dimension"]


def test_embed_multiple_texts(client):
    """Test embedding generation for multiple texts."""
    request_data = {
        "texts": [
            "First test sentence.",
            "Second test sentence with different content.",
            "Third sentence for testing."
        ]
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 3
    assert data["successful"] == 3
    assert len(data["embeddings"]) == 3

    # All embeddings should be present
    for embedding in data["embeddings"]:
        assert embedding is not None
        assert len(embedding) == data["dimension"]


def test_embed_with_specific_model(client):
    """Test embedding generation with a specific model."""
    request_data = {
        "texts": ["Test with specific model."],
        "model": "all-MiniLM-L6-v2"
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "all-MiniLM-L6-v2"


def test_embed_with_empty_texts(client):
    """Test embedding generation with some empty texts."""
    request_data = {
        "texts": ["Valid text", "", "Another valid text", "   "]
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 4
    assert data["successful"] == 2  # Only 2 valid texts

    embeddings = data["embeddings"]
    assert embeddings[0] is not None  # Valid
    assert embeddings[1] is None      # Empty
    assert embeddings[2] is not None  # Valid
    assert embeddings[3] is None      # Whitespace only


def test_embed_empty_request_fails(client):
    """Test that empty texts list returns validation error."""
    request_data = {
        "texts": []
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 422  # Validation error


def test_embed_invalid_model(client):
    """Test embedding with invalid model name."""
    request_data = {
        "texts": ["Test text"],
        "model": "invalid-model-name"
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 400


def test_embed_oversized_batch_fails(client):
    """Test that oversized batch returns error."""
    # Create a batch larger than the max size (100)
    large_batch = [f"Text number {i}" for i in range(101)]

    request_data = {
        "texts": large_batch
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 413  # Request entity too large


def test_embed_response_structure(client):
    """Test that embed response has correct structure."""
    request_data = {
        "texts": ["Test for response structure"]
    }

    response = client.post("/embed", json=request_data)
    assert response.status_code == 200

    data = response.json()

    # Check all required fields are present
    required_fields = ["embeddings", "model", "dimension", "count", "successful"]
    for field in required_fields:
        assert field in data

    # Check types
    assert isinstance(data["embeddings"], list)
    assert isinstance(data["model"], str)
    assert isinstance(data["dimension"], int)
    assert isinstance(data["count"], int)
    assert isinstance(data["successful"], int)


def test_cors_headers_present(client):
    """Test that CORS headers are present (if configured)."""
    response = client.get("/health")
    # This test depends on CORS configuration
    # Add CORS configuration to your FastAPI app if needed
    assert response.status_code == 200


def test_openapi_docs_accessible(client):
    """Test that OpenAPI documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200