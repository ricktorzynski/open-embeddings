FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy application code
COPY open_embeddings/ ./open_embeddings/

# Create non-root user
RUN useradd --create-home --shell /bin/bash embedding
RUN chown -R embedding:embedding /app
USER embedding

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8765/health')"

# Run the application
CMD ["python", "-m", "open_embeddings.main"]