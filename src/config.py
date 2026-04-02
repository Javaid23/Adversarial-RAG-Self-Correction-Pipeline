from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

from dotenv import load_dotenv

# Find project root by looking for the data directory
def get_project_root() -> Path:
    """Get the project root directory by searching for the data directory."""
    # Start from current working directory
    current = Path.cwd().resolve()
    
    # Check current directory first
    if (current / "data" / "docs" / "adversarial").exists():
        return current
    
    # Check if a subdirectory contains what we need
    for subdir in current.iterdir():
        if subdir.is_dir() and (subdir / "data" / "docs" / "adversarial").exists():
            return subdir
    
    # Check parent directories (up to 3 levels)
    for _ in range(3):
        if (current / "data" / "docs" / "adversarial").exists():
            return current
        current = current.parent
    
    # Fallback: return cwd
    return Path.cwd().resolve()


load_dotenv()
PROJECT_ROOT = get_project_root()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    victim_model: str = "llama3:8b"
    embedding_model: str = "nomic-embed-text"
    vector_db: str = "chroma"
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    docs_dir: Path = None  # Set in get_settings()
    index_dir: Path = None  # Set in get_settings()

    def __post_init__(self):
        if self.docs_dir is None:
            object.__setattr__(self, 'docs_dir', PROJECT_ROOT / "data" / "docs")
        if self.index_dir is None:
            object.__setattr__(self, 'index_dir', PROJECT_ROOT / "data" / "index")


def get_settings() -> Settings:
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        victim_model=os.getenv("VICTIM_MODEL", "llama3:8b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        vector_db=os.getenv("VECTOR_DB", "chroma"),
        top_k=int(os.getenv("TOP_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        docs_dir=PROJECT_ROOT / "data" / "docs",
        index_dir=PROJECT_ROOT / "data" / "index",
    )
