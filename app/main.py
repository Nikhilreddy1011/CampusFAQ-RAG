import logging
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import (
    BASE_DIR, SAMPLE_DOCS_DIR, UPLOAD_DIR,
    SIMILARITY_THRESHOLD, TOP_K, LOG_RAW_QUERIES
)
from app.database import (
    init_db, get_all_documents, get_document_by_id,
    get_document_by_hash, insert_document_and_chunks,
    delete_document, get_chunks_by_document, get_stats
)
from app.document_processor import process_document, compute_sha256
from app.embeddings import embedding_service
from app.retriever import retrieve_relevant_chunks
from app.answer_generator import generate_grounded_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("campus_faq.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database and preload embedding model on startup."""
    logger.info("Initializing Campus FAQ RAG System...")
    init_db()
    # Eagerly load the FastEmbed model so student queries are snappy
    embedding_service.embed_texts(["System warm up."])
    logger.info("System ready.")
    yield


app = FastAPI(
    title="Campus FAQ Retrieval-Augmented Generation System",
    description="A grounded, privacy-respecting campus helpdesk RAG assistant.",
    version="1.0.0",
    lifespan=lifespan
)

# Static files and Templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


# --- Pydantic Request Models ---
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Student question")
    threshold: Optional[float] = Field(default=SIMILARITY_THRESHOLD, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=TOP_K, ge=1, le=10)


# --- HTML Page Routes ---
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the primary Student Helpdesk chat interface."""
    stats = get_stats()
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"stats": stats}
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Serve the Admin Document Management portal."""
    docs = get_all_documents()
    stats = get_stats()
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "documents": docs,
            "stats": stats,
            "default_threshold": SIMILARITY_THRESHOLD
        }
    )


# --- API Routes ---
@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Core RAG question-answering endpoint.
    Retrieves top relevant chunks, tests similarity against threshold,
    and returns a grounded answer with citations or fallback routing.
    """
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t0 = time.perf_counter()
    
    # Semantic vector retrieval
    retrieval = retrieve_relevant_chunks(
        query=clean_query,
        top_k=payload.top_k,
        threshold=payload.threshold
    )

    # Grounded answer synthesis
    answer_payload = generate_grounded_answer(clean_query, retrieval)
    
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Privacy-conscious logging
    if LOG_RAW_QUERIES:
        logger.info("Chat Query: '%s' | Grounded: %s | Score: %.3f | Latency: %.1fms",
                    clean_query, answer_payload["grounded"], answer_payload["confidence_score"], elapsed_ms)
    else:
        logger.info("Chat Query processed | Length: %d chars | Grounded: %s | Score: %.3f | Latency: %.1fms",
                    len(clean_query), answer_payload["grounded"], answer_payload["confidence_score"], elapsed_ms)

    return {
        "query": clean_query,
        "answer": answer_payload["answer"],
        "grounded": answer_payload["grounded"],
        "confidence_score": answer_payload["confidence_score"],
        "citations": answer_payload["citations"],
        "suggested_office": answer_payload["suggested_office"],
        "search_time_ms": elapsed_ms,
        "total_chunks_indexed": retrieval["total_chunks_indexed"],
        "synthesizer": answer_payload.get("synthesizer", "extractive")
    }


@app.get("/api/documents")
async def list_documents():
    """List all currently indexed campus documents."""
    docs = get_all_documents()
    return {"documents": docs, "total": len(docs)}


@app.get("/api/documents/{doc_id}/chunks")
async def view_document_chunks(doc_id: int):
    """Retrieve indexed chunks for a given document for inspection."""
    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    chunks = get_chunks_by_document(doc_id)
    return {"document": doc, "chunks": chunks}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Admin endpoint to upload and index a campus document (.pdf, .docx, .txt).
    Deduplicates by SHA-256 hash.
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only .pdf, .docx, and .txt files are allowed."
        )

    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Deduplication check
    file_hash = compute_sha256(content_bytes)
    existing_doc = get_document_by_hash(file_hash)
    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail=f"This document is already indexed as '{existing_doc['filename']}' (uploaded {existing_doc['uploaded_at']})."
        )

    # Process and chunk document
    try:
        chunks, file_type = process_document(filename, content_bytes)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Error processing document %s", filename)
        raise HTTPException(status_code=500, detail=f"Document parsing failed: {str(e)}")

    # Embed chunks using FastEmbed
    chunk_texts = [c["content"] for c in chunks]
    try:
        embeddings = embedding_service.embed_texts(chunk_texts)
    except Exception as e:
        logger.exception("Error embedding chunks for %s", filename)
        raise HTTPException(status_code=500, detail=f"Vector embedding failed: {str(e)}")

    # Save original file to upload directory
    save_path = UPLOAD_DIR / f"{file_hash[:10]}_{filename}"
    try:
        save_path.write_bytes(content_bytes)
    except Exception as e:
        logger.warning("Could not persist original file to disk: %s", e)

    # Insert into database
    doc_id = insert_document_and_chunks(
        filename=filename,
        file_type=file_type,
        file_size=len(content_bytes),
        sha256_hash=file_hash,
        chunks=chunks,
        embeddings=embeddings
    )

    logger.info("Successfully indexed '%s' (ID: %d, Chunks: %d)", filename, doc_id, len(chunks))

    return {
        "message": f"Successfully indexed '{filename}'.",
        "document_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "chunks_indexed": len(chunks)
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document_endpoint(doc_id: int):
    """Admin endpoint to delete a document and all of its associated chunks."""
    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    success = delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document from database.")

    logger.info("Deleted document ID %d ('%s')", doc_id, doc["filename"])
    return {"message": f"Document '{doc['filename']}' and its indexed chunks were deleted."}


@app.post("/api/admin/load-samples")
async def load_sample_documents():
    """Convenience endpoint to load and index all provided sample campus FAQs."""
    if not SAMPLE_DOCS_DIR.exists():
        raise HTTPException(status_code=404, detail="Sample documents directory not found.")

    sample_files = list(SAMPLE_DOCS_DIR.glob("*.*"))
    if not sample_files:
        raise HTTPException(status_code=404, detail="No sample documents found.")

    loaded = []
    skipped = []
    errors = []

    for path in sample_files:
        if path.suffix.lower() not in [".pdf", ".docx", ".txt"]:
            continue
        try:
            content_bytes = path.read_bytes()
            file_hash = compute_sha256(content_bytes)

            # Check if already indexed
            if get_document_by_hash(file_hash):
                skipped.append(path.name)
                continue

            chunks, file_type = process_document(path.name, content_bytes)
            chunk_texts = [c["content"] for c in chunks]
            embeddings = embedding_service.embed_texts(chunk_texts)

            doc_id = insert_document_and_chunks(
                filename=path.name,
                file_type=file_type,
                file_size=len(content_bytes),
                sha256_hash=file_hash,
                chunks=chunks,
                embeddings=embeddings
            )
            loaded.append({"filename": path.name, "chunks": len(chunks), "doc_id": doc_id})
        except Exception as e:
            logger.exception("Failed loading sample doc %s", path.name)
            errors.append({"filename": path.name, "error": str(e)})

    return {
        "message": f"Sample ingestion complete. Added: {len(loaded)}, Already existed: {len(skipped)}, Errors: {len(errors)}",
        "loaded": loaded,
        "skipped": skipped,
        "errors": errors
    }


@app.get("/api/stats")
async def stats_endpoint():
    """Return database statistics and system settings."""
    stats = get_stats()
    stats["similarity_threshold"] = SIMILARITY_THRESHOLD
    stats["top_k"] = TOP_K
    return stats
