import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from app.config import TOP_K, SIMILARITY_THRESHOLD
from app.database import get_all_chunks_with_embeddings
from app.embeddings import embedding_service

logger = logging.getLogger("campus_faq.retriever")

def retrieve_relevant_chunks(
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """
    Perform semantic vector search against all stored FAQ chunks in SQLite.
    
    Returns a dictionary containing:
      - 'chunks': List of matching chunks sorted by similarity score
      - 'top_score': float highest similarity score
      - 'passed_threshold': bool whether top result meets the cutoff
      - 'search_time_ms': float execution duration
      - 'total_chunks_indexed': int total chunks evaluated
    """
    start_time = time.perf_counter()
    
    # 1. Fetch chunks from SQLite
    stored_chunks = get_all_chunks_with_embeddings()
    if not stored_chunks:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return {
            "chunks": [],
            "top_score": 0.0,
            "passed_threshold": False,
            "search_time_ms": round(duration_ms, 2),
            "total_chunks_indexed": 0
        }

    # 2. Embed user question
    query_vector = embedding_service.embed_query(query)

    # 3. Stack chunk embeddings into matrix
    chunk_vectors = np.stack([c["embedding"] for c in stored_chunks])

    # 4. Compute cosine similarities (dot product of unit vectors)
    similarities = embedding_service.compute_cosine_similarity(query_vector, chunk_vectors)

    # 5. Attach similarity score to each chunk candidate
    scored_items = []
    for idx, sim in enumerate(similarities):
        score_val = float(sim)
        item = {
            "chunk_id": stored_chunks[idx]["chunk_id"],
            "doc_id": stored_chunks[idx]["doc_id"],
            "filename": stored_chunks[idx]["filename"],
            "chunk_index": stored_chunks[idx]["chunk_index"],
            "content": stored_chunks[idx]["content"],
            "page_number": stored_chunks[idx]["page_number"],
            "section_heading": stored_chunks[idx]["section_heading"],
            "similarity_score": round(score_val, 4)
        }
        scored_items.append((score_val, item))

    # Sort descending by similarity
    scored_items.sort(key=lambda x: x[0], reverse=True)

    top_score = scored_items[0][0] if scored_items else 0.0
    passed_threshold = (top_score >= threshold)

    # Take top_k results that meet the threshold
    qualifying_chunks = []
    if passed_threshold:
        for score, chunk in scored_items[:top_k]:
            if score >= threshold:
                qualifying_chunks.append(chunk)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Semantic search completed in %.2f ms | Query length: %d chars | Total indexed: %d | Top score: %.4f | Retained: %d",
        duration_ms, len(query), len(stored_chunks), top_score, len(qualifying_chunks)
    )

    return {
        "chunks": qualifying_chunks,
        "top_score": round(top_score, 4),
        "passed_threshold": passed_threshold,
        "search_time_ms": round(duration_ms, 2),
        "total_chunks_indexed": len(stored_chunks)
    }
