import pytest
from app.database import insert_document_and_chunks, delete_document, get_all_documents
from app.embeddings import embedding_service
from app.retriever import retrieve_relevant_chunks

def test_retrieval_empty_db():
    result = retrieve_relevant_chunks("What are the clinic hours?")
    assert result["chunks"] == []
    assert result["top_score"] == 0.0
    assert result["passed_threshold"] is False
    assert result["total_chunks_indexed"] == 0

def test_semantic_retrieval_hit():
    # Ingest a sample doc
    chunks = [
        {
            "chunk_index": 0,
            "content": "Students may add or drop full-term courses via the Student Portal through September 15.",
            "page_number": 1,
            "section_heading": "Course Registration"
        },
        {
            "chunk_index": 1,
            "content": "The campus health clinic is open Monday through Friday from 8:00 AM to 5:00 PM in Building 4.",
            "page_number": 1,
            "section_heading": "Health Services"
        }
    ]
    texts = [c["content"] for c in chunks]
    embeddings = embedding_service.embed_texts(texts)

    doc_id = insert_document_and_chunks(
        filename="test_policy.txt",
        file_type="TXT",
        file_size=500,
        sha256_hash="test_sha256_dummy_1",
        chunks=chunks,
        embeddings=embeddings
    )

    # Search for health clinic
    result = retrieve_relevant_chunks("What time does the health clinic open?", top_k=2)
    assert result["passed_threshold"] is True
    assert len(result["chunks"]) >= 1
    top_chunk = result["chunks"][0]
    assert "health clinic is open" in top_chunk["content"]
    assert top_chunk["similarity_score"] >= 0.50

def test_irrelevant_query_threshold():
    chunks = [{
        "chunk_index": 0,
        "content": "Students may add or drop full-term courses via the Student Portal through September 15.",
        "page_number": 1,
        "section_heading": "Course Registration"
    }]
    texts = [c["content"] for c in chunks]
    embeddings = embedding_service.embed_texts(texts)

    insert_document_and_chunks(
        filename="registration.txt",
        file_type="TXT",
        file_size=200,
        sha256_hash="hash_registration_irrel_test",
        chunks=chunks,
        embeddings=embeddings
    )

    # Query completely outside university policy
    result = retrieve_relevant_chunks(
        "What is the best recipe to bake chocolate chip brownies with walnuts?",
        threshold=0.45
    )
    # Cosine similarity for brownies against registration is ~0.36, below 0.45 threshold
    assert result["passed_threshold"] is False
    assert len(result["chunks"]) == 0

def test_document_deletion_cascades_retrieval():
    chunks = [{
        "chunk_index": 0,
        "content": "Student commuter parking permits cost $175 per semester for the Yellow Lot.",
        "page_number": 1,
        "section_heading": "Parking Rates"
    }]
    texts = [c["content"] for c in chunks]
    embeddings = embedding_service.embed_texts(texts)

    doc_id = insert_document_and_chunks(
        filename="parking.txt",
        file_type="TXT",
        file_size=200,
        sha256_hash="hash_parking_delete_test",
        chunks=chunks,
        embeddings=embeddings
    )

    # Retrieval works before deletion
    res_before = retrieve_relevant_chunks("How much is a parking permit?")
    assert len(res_before["chunks"]) >= 1

    # Delete the document
    deleted = delete_document(doc_id)
    assert deleted is True

    # Retrieval should now find nothing
    res_after = retrieve_relevant_chunks("How much is a parking permit?")
    assert len(res_after["chunks"]) == 0
    assert res_after["total_chunks_indexed"] == 0
