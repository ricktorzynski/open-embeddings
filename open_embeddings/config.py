"""Configuration management for the embedding server."""

import os
from typing import Optional


class Config:
    """Server configuration."""

    def __init__(self):
        self.host = os.getenv("EMBEDDING_HOST", "0.0.0.0")
        self.port = int(os.getenv("EMBEDDING_PORT", "8765"))
        self.default_model = os.getenv("DEFAULT_MODEL", "all-MiniLM-L6-v2")
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "100"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.workers = int(os.getenv("WORKERS", "1"))

    @property
    def uvicorn_config(self) -> dict:
        """Get configuration dict for uvicorn."""
        return {
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level.lower(),
            "workers": self.workers,
        }


# Global config instance
config = Config()