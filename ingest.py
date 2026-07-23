"""Ingest pipeline: load corpus files -> chunk -> embed locally -> store in Chroma.

Same pipeline as Week 3 (corpus, chunking, and embedding model are unchanged) --
here the resulting collection is queried by the agent's search_knowledge_base
tool (see rag_tool.py) instead of a standalone interactive Q&A script.

Reads both .md (plain text) and .pdf (extracted via pypdf) files from the
corpus directory, so the same pipeline can be pointed at a PDF corpus (e.g.
corpus_pdf/, see generate_sample_pdfs.py) to compare extraction quality
against the original markdown.

Run this once (or whenever the corpus directory changes) before running agent.py:
    python ingest.py               # ingests corpus/ (default)
    python ingest.py corpus_pdf    # ingests a different directory instead
"""
import pathlib
import sys

import chromadb
import pypdf
from chromadb.utils import embedding_functions

from chunking import semantic_chunk_text

CORPUS_DIR = pathlib.Path(__file__).parent / "corpus"
DB_DIR = pathlib.Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "nimbus_docs"

# Runs entirely on-device, no API key or network calls needed.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _read_text(path: pathlib.Path) -> str:
    """Read a corpus file's text, extracting it from PDF pages if needed."""
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def build_collection(corpus_dir: pathlib.Path = CORPUS_DIR) -> chromadb.Collection:
    """Rebuild the Chroma collection from scratch: read every .md/.pdf file in
    corpus_dir, chunk it, embed each chunk locally, and store it with its
    source filename as metadata."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Fresh collection every run so re-ingesting doesn't duplicate or leave stale chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    ids, documents, metadatas = [], [], []
    paths = sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.pdf"))
    for path in paths:
        text = _read_text(path)
        for i, chunk in enumerate(semantic_chunk_text(text)):
            ids.append(f"{path.stem}::{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i, "status": "current"})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def main() -> None:
    """Entry point for `python ingest.py [corpus_dir]` — builds the collection
    and reports how many chunks landed in it. Defaults to corpus/ if no
    directory is given on the command line."""
    corpus_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else CORPUS_DIR
    collection = build_collection(corpus_dir)
    print(f"Ingested {collection.count()} chunks from {corpus_dir} into {DB_DIR}")


if __name__ == "__main__":
    main()
