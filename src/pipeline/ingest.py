from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from langchain_community.vectorstores import Chroma

from src.llm.provider import OllamaProvider


@dataclass
class Chunk:
    id: int
    source: str
    text: str


def _read_text_files(docs_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    if not docs_dir.exists():
        return docs

    for file_path in docs_dir.rglob("*.txt"):
        docs.append((str(file_path), file_path.read_text(encoding="utf-8", errors="ignore")))
    return docs


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def build_index(
    provider: OllamaProvider,
    docs_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    # Ensure paths are Path objects
    docs_dir = Path(docs_dir) if not isinstance(docs_dir, Path) else docs_dir
    index_dir = Path(index_dir) if not isinstance(index_dir, Path) else index_dir
    
    docs = _read_text_files(docs_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    persist_dir = index_dir / "chroma"

    all_chunks: list[Chunk] = []
    chunk_id = 0

    for source, text in docs:
        for c in _chunk_text(text, chunk_size, chunk_overlap):
            all_chunks.append(Chunk(id=chunk_id, source=source, text=c))
            chunk_id += 1

    if not all_chunks:
        raise ValueError(
            f"No .txt documents found in {docs_dir}. Add files to data/docs and retry."
        )

    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    texts = [c.text for c in all_chunks]
    metadatas = [{"id": c.id, "source": c.source} for c in all_chunks]

    Chroma.from_texts(
        texts=texts,
        embedding=provider.get_embedding_function(),
        metadatas=metadatas,
        persist_directory=str(persist_dir),
        collection_name="rag_chunks",
    )

    return len(all_chunks)
