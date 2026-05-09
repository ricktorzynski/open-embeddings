# Open Embeddings

A lightweight, fast, and focused HTTP API server for generating text embeddings using sentence-transformers.

## Features

- **Single Purpose**: Dedicated to text embeddings only (no LLM mixing)
- **Multiple Models**: Support for various sentence-transformer models
- **Batch Processing**: Efficient batch embedding generation
- **Normalized Output**: All embeddings normalized to unit length for cosine similarity
- **Health Checks**: Built-in health and readiness endpoints
- **Fast**: Optimized for embedding generation performance
- **Docker Ready**: Containerized deployment support

## Why This Exists

Most embedding services are bundled with LLM APIs (OpenAI, Anthropic) or mixed-purpose servers (Ollama). This server provides:

- Dedicated embedding generation without LLM overhead
- Local deployment without external API dependencies
- Focused, minimal API surface
- High performance for vector search applications

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
uv run python -m open_embeddings

# Test with curl
curl -X POST http://localhost:8765/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "This is a test"]}'
```

## API Endpoints

### POST /embed
Generate embeddings for one or more texts.

**Request:**
```json
{
  "texts": ["text1", "text2"],
  "model": "all-MiniLM-L6-v2"  // optional
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "model": "all-MiniLM-L6-v2",
  "dimension": 384
}
```

### GET /health
Health check endpoint.

### GET /models
List available models.

## Configuration

Environment variables:
- `EMBEDDING_HOST`: Server host (default: 0.0.0.0)
- `EMBEDDING_PORT`: Server port (default: 8765)
- `DEFAULT_MODEL`: Default embedding model (default: all-MiniLM-L6-v2)
- `MAX_BATCH_SIZE`: Maximum texts per request (default: 100)

## Docker

```bash
docker build -t open-embeddings .
docker run -p 8765:8765 open-embeddings
```

## License

MIT