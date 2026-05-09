"""Main entry point for the embedding server."""

import uvicorn
from .app import app
from .config import config


def main():
    """Run the embedding server."""
    uvicorn.run(
        "open_embeddings.app:app",
        **config.uvicorn_config
    )


if __name__ == "__main__":
    main()