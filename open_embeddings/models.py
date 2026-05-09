"""API request and response models."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,  # Configurable batch size limit
        description="List of texts to embed"
    )
    model: Optional[str] = Field(
        None,
        description="Model name to use (optional, uses default if not specified)"
    )

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate that texts list is not empty and contains valid strings."""
        if not v:
            raise ValueError("texts cannot be empty")

        # Allow empty strings - they'll be handled by the embedding generator
        return v


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""

    embeddings: List[Optional[List[float]]] = Field(
        ...,
        description="List of embedding vectors (null for failed texts)"
    )
    model: str = Field(
        ...,
        description="Model used for embedding generation"
    )
    dimension: int = Field(
        ...,
        description="Dimension of the embedding vectors"
    )
    count: int = Field(
        ...,
        description="Number of texts processed"
    )
    successful: int = Field(
        ...,
        description="Number of successfully processed texts"
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Server version")


class ModelsResponse(BaseModel):
    """Available models response model."""

    models: List[str] = Field(..., description="List of available model names")
    default: str = Field(..., description="Default model name")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")