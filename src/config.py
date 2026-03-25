from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    victim_model: str = "llama3:8b"
    embedding_model: str = "nomic-embed-text"
    vector_db: str = "chroma"
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")


def get_settings() -> Settings:
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        victim_model=os.getenv("VICTIM_MODEL", "llama3:8b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        vector_db=os.getenv("VECTOR_DB", "chroma"),
        top_k=int(os.getenv("TOP_K", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
    )
