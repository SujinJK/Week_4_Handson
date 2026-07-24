"""The Week 3 RAG retriever, trimmed down to one function the agent calls as a tool.

Same retrieval pipeline validated in Week 3 (hybrid vector+keyword search via
Reciprocal Rank Fusion, then cross-encoder reranking) -- generation and the
interactive trace-printing are dropped since here the agent (agent.py), not
this module, is the one calling Claude and deciding what to show the user.

For readers new to RAG, the four ideas below are used throughout this file:

- "Embedding": converting a chunk of text into a list of numbers (a
  vector) that represents its *meaning*. Two chunks about similar topics
  end up with similar numbers, even if they don't share exact words --
  that's what makes "vector search" possible.
- "Vector search" (`_vector_search`): given the question's own embedding,
  find the stored chunks whose embeddings are numerically closest to it.
  Good at matching meaning, weaker at matching exact rare terms.
- "BM25" (`_bm25_search`): a classic keyword-matching algorithm (no
  embeddings involved) that scores chunks by how well their actual words
  overlap with the question's words. Good at exact terms, blind to
  synonyms/meaning.
- "Reciprocal Rank Fusion" / RRF (`_hybrid_retrieve`) and "reranking"
  (`_rerank`): vector search and BM25 each produce their own ranked list
  for the same question; RRF is just a simple formula for merging two
  ranked lists into one without needing their scores to be on the same
  scale. Reranking then takes that merged shortlist and re-scores it with
  a slower but more accurate model (a "cross-encoder") that looks at the
  question and each chunk together, to pick the very best few.
"""
import pathlib
import re

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingest import COLLECTION_NAME, DB_DIR, EMBEDDING_MODEL

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_K = 8  # wider candidate pool handed to the reranker
FINAL_K = 3    # how many chunks actually reach Claude, after reranking
RRF_K = 60     # standard damping constant for Reciprocal Rank Fusion

_collection: chromadb.Collection | None = None
_reranker: CrossEncoder | None = None
_bm25_index_cache: dict[int, tuple[BM25Okapi, list[dict]]] = {}
_TOKEN_RE = re.compile(r"\w+")


def get_collection() -> chromadb.Collection:
    """Open (and cache) the Chroma collection built by ingest.py."""
    global _collection
    if _collection is None:
        if not pathlib.Path(DB_DIR).exists():
            raise RuntimeError("No vector store found. Run `python ingest.py` first.")
        client = chromadb.PersistentClient(path=str(DB_DIR))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    return _collection


def _vector_search(collection: chromadb.Collection, question: str, k: int) -> list[dict]:
    results = collection.query(query_texts=[question], n_results=k, where={"status": "current"})
    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": text, "source": meta["source"], "chunk_index": meta["chunk_index"], "distance": distance})
    return hits


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _get_bm25_index(collection: chromadb.Collection) -> tuple[BM25Okapi, list[dict]]:
    cache_key = id(collection)
    if cache_key not in _bm25_index_cache:
        result = collection.get(where={"status": "current"})
        chunks = [
            {"text": text, "source": meta["source"], "chunk_index": meta["chunk_index"]}
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        _bm25_index_cache[cache_key] = (BM25Okapi(tokenized_corpus), chunks)
    return _bm25_index_cache[cache_key]


def _bm25_search(collection: chromadb.Collection, question: str, k: int) -> list[dict]:
    bm25, chunks = _get_bm25_index(collection)
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [{**chunk, "bm25_score": float(score)} for chunk, score in ranked[:k]]


def _hybrid_retrieve(collection: chromadb.Collection, question: str, k: int = INITIAL_K) -> list[dict]:
    vector_hits = _vector_search(collection, question, k=k)
    keyword_hits = _bm25_search(collection, question, k=k)

    fused_scores: dict[tuple, float] = {}
    chunk_lookup: dict[tuple, dict] = {}
    for ranked_list in (vector_hits, keyword_hits):
        for rank, hit in enumerate(ranked_list):
            key = (hit["source"], hit["chunk_index"])
            # RRF's whole trick: instead of combining vector distances and BM25
            # scores directly (they're on totally different scales), only look
            # at each chunk's *position* (rank) in each list. A chunk ranked
            # #1 in either list scores 1/(RRF_K+1); #2 scores 1/(RRF_K+2); and
            # so on -- so a chunk that ranks highly in *both* lists accumulates
            # the most points once we sum across both loops.
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_lookup[key] = {**chunk_lookup.get(key, {}), **hit}

    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    return [chunk_lookup[key] for key in ranked_keys[:k]]


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _rerank(question: str, hits: list[dict], top_n: int = FINAL_K) -> list[dict]:
    reranker = _get_reranker()
    pairs = [(question, hit["text"]) for hit in hits]
    scores = reranker.predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_n]


def search_knowledge_base(query: str) -> str:
    """Hybrid search + rerank over the Nimbus corpus, formatted as numbered,
    source-labeled snippets the agent can cite by number."""
    collection = get_collection()
    candidates = _hybrid_retrieve(collection, query, k=INITIAL_K)
    hits = _rerank(query, candidates, top_n=FINAL_K)
    if not hits:
        return "No matching documents found in the knowledge base."
    parts = [f"[{i}] (source: {hit['source']})\n{hit['text']}" for i, hit in enumerate(hits, start=1)]
    return "\n\n".join(parts)
