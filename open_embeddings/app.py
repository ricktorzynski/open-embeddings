"""FastAPI application for the embedding server."""

import logging
from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from . import __version__
from .config import config
from .embeddings import get_embedding_generator, get_available_models
from .models import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelsResponse,
    ErrorResponse
)


# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper()))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Open Embeddings",
    description="A lightweight HTTP API server for text embeddings",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available embedding models."""
    return ModelsResponse(
        models=get_available_models(),
        default=config.default_model
    )


@app.post("/embed", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    """
    Generate embeddings for the provided texts.

    Returns normalized embedding vectors suitable for cosine similarity.
    """
    try:
        # Validate batch size
        if len(request.texts) > config.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Batch size {len(request.texts)} exceeds maximum {config.max_batch_size}"
            )

        # Determine model to use
        model_name = request.model or config.default_model

        # Get embedding generator
        try:
            generator = get_embedding_generator(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load model '{model_name}': {str(e)}"
            )

        # Generate embeddings
        embeddings = generator.embed_batch(request.texts)

        # Convert numpy arrays to lists and count successes
        embedding_lists = []
        successful_count = 0

        for embedding in embeddings:
            if embedding is not None:
                # Convert numpy array to Python list
                embedding_lists.append(embedding.tolist())
                successful_count += 1
            else:
                embedding_lists.append(None)

        # Create response
        return EmbedResponse(
            embeddings=embedding_lists,
            model=model_name,
            dimension=generator.dimension,
            count=len(request.texts),
            successful=successful_count
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=getattr(exc, 'detail', None)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if config.log_level.upper() == "DEBUG" else None
        ).model_dump()
    )


# Add startup event to log configuration
@app.on_event("startup")
async def startup_event():
    """Log server startup information."""
    logger.info(f"Starting Open Embeddings v{__version__}")
    logger.info(f"Default model: {config.default_model}")
    logger.info(f"Max batch size: {config.max_batch_size}")
    logger.info(f"Available models: {', '.join(get_available_models())}")


# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Log server shutdown."""
    logger.info("Open Embeddings shutting down")