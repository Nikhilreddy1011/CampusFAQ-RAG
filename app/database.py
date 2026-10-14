import sqlite3
from typing import List, Dict, Any, Optional
import numpy as np
import app.config as config

def get_connection() -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database with WAL mode."""
    conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

def init_db():
    """Initialize database tables and indices if they do not already exist."""
    conn = get_connection()
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    sha256 TEXT UNIQUE NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    section_heading TEXT,
                    embedding BLOB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_docs_sha256 ON documents(sha256);
            """)
    finally:
        conn.close()

def get_document_by_hash(sha256_hash: str) -> Optional[Dict[str, Any]]:
    """Check if a document with identical content hash already exists."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256_hash,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve document metadata by primary key ID."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_all_documents() -> List[Dict[str, Any]]:
    """Return all indexed documents sorted by upload date descending."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, filename, file_type, file_size, sha256, chunk_count, uploaded_at 
            FROM documents 
            ORDER BY uploaded_at DESC
        """)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def insert_document_and_chunks(
    filename: str,
    file_type: str,
    file_size: int,
    sha256_hash: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray
) -> int:
    """
    Atomically insert document metadata and all generated chunks with their embeddings.
    embeddings is expected to be a float32 2D numpy array of shape (len(chunks), embedding_dim).
    """
    conn = get_connection()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO documents (filename, file_type, file_size, sha256, chunk_count)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, file_type, file_size, sha256_hash, len(chunks)))
            doc_id = cur.lastrowid

            chunk_records = []
            for i, chunk in enumerate(chunks):
                emb_bytes = embeddings[i].astype(np.float32).tobytes()
                chunk_records.append((
                    doc_id,
                    chunk.get("chunk_index", i),
                    chunk["content"],
                    chunk.get("page_number", 1),
                    chunk.get("section_heading", "General"),
                    emb_bytes
                ))

            cur.executemany("""
                INSERT INTO chunks (doc_id, chunk_index, content, page_number, section_heading, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, chunk_records)

            return doc_id
    finally:
        conn.close()

def delete_document(doc_id: int) -> bool:
    """Delete a document and cascade delete all its indexed chunks."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cur.rowcount > 0
    finally:
        conn.close()

def get_all_chunks_with_embeddings() -> List[Dict[str, Any]]:
    """
    Load all chunks joined with document filenames for retrieval similarity computation.
    Decodes stored BLOBs back into numpy float32 arrays.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                c.id AS chunk_id,
                c.doc_id,
                d.filename,
                c.chunk_index,
                c.content,
                c.page_number,
                c.section_heading,
                c.embedding
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
        """)
        rows = cur.fetchall()
        results = []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            results.append({
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "page_number": r["page_number"],
                "section_heading": r["section_heading"],
                "embedding": emb
            })
        return results
    finally:
        conn.close()

def get_chunks_by_document(doc_id: int) -> List[Dict[str, Any]]:
    """Retrieve all chunks for a specific document (without raw embedding blobs)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, chunk_index, content, page_number, section_heading
            FROM chunks
            WHERE doc_id = ?
            ORDER BY chunk_index ASC
        """, (doc_id,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def get_stats() -> Dict[str, Any]:
    """Return summary statistics of the document database."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        total_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cur.fetchone()[0]
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "db_path": str(config.DB_PATH)
        }
    finally:
        conn.close()
