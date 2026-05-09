#!/usr/bin/env python3
"""Example client for the embedding server."""

import requests
import json
import numpy as np


class EmbeddingClient:
    """Simple client for the embedding server."""

    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url

    def health_check(self):
        """Check if the server is healthy."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self):
        """Get list of available models."""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def embed(self, texts, model=None):
        """Generate embeddings for texts."""
        payload = {"texts": texts}
        if model:
            payload["model"] = model

        response = requests.post(
            f"{self.base_url}/embed",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        # Convert to numpy arrays if they aren't already
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)

        # Since embeddings are normalized, dot product = cosine similarity
        return np.dot(embedding1, embedding2)


def main():
    """Demonstrate the embedding client."""
    # Initialize client
    client = EmbeddingClient()

    # Check server health
    print("Checking server health...")
    health = client.health_check()
    print(f"Server status: {health['status']} (version: {health['version']})")

    # List available models
    print("\nListing available models...")
    models_info = client.list_models()
    print(f"Default model: {models_info['default']}")
    print(f"Available models: {', '.join(models_info['models'])}")

    # Example texts to embed
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown animal leaps over a sleepy canine",
        "Python is a programming language",
        "Machine learning is fascinating"
    ]

    print(f"\nGenerating embeddings for {len(texts)} texts...")
    result = client.embed(texts)

    print(f"Model used: {result['model']}")
    print(f"Embedding dimension: {result['dimension']}")
    print(f"Texts processed: {result['count']}")
    print(f"Successful embeddings: {result['successful']}")

    # Calculate similarities between first two texts (should be high)
    if len(result['embeddings']) >= 2 and all(emb is not None for emb in result['embeddings'][:2]):
        emb1 = result['embeddings'][0]
        emb2 = result['embeddings'][1]
        similarity = client.cosine_similarity(emb1, emb2)
        print(f"\nSimilarity between first two texts: {similarity:.4f}")

    # Calculate similarity between semantically different texts
    if len(result['embeddings']) >= 4 and all(emb is not None for emb in [result['embeddings'][0], result['embeddings'][2]]):
        emb1 = result['embeddings'][0]  # Fox sentence
        emb3 = result['embeddings'][2]  # Python sentence
        similarity = client.cosine_similarity(emb1, emb3)
        print(f"Similarity between different topics: {similarity:.4f}")

    # Example with specific model
    print(f"\nTesting with specific model...")
    try:
        result_specific = client.embed(
            ["Test with specific model"],
            model="all-MiniLM-L6-v2"
        )
        print(f"Used model: {result_specific['model']}")
        print(f"Embedding dimension: {result_specific['dimension']}")
    except requests.exceptions.HTTPError as e:
        print(f"Error with specific model: {e}")

    print("\nClient example completed!")


if __name__ == "__main__":
    main()