import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_all_documents

client = TestClient(app)

def test_homepage_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "Campus FAQ System" in res.text
    assert "Ask Campus Services" in res.text

def test_admin_page_endpoint():
    res = client.get("/admin")
    assert res.status_code == 200
    assert "Document Management Portal" in res.text

def test_stats_endpoint():
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "similarity_threshold" in data

def test_upload_and_chat_flow():
    # 1. Upload a test policy
    file_content = (
        "SECTION: Student ID Card Replacement\n"
        "Students who lose their campus ID card can obtain a replacement card at the Card Services Desk "
        "in the Student Union, Room 104. The replacement fee is $25, payable via credit card or student account debit."
    ).encode("utf-8")

    files = {"file": ("id_card_policy.txt", file_content, "text/plain")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["filename"] == "id_card_policy.txt"
    doc_id = upload_data["document_id"]

    # 2. Duplicate upload should return 409 Conflict
    dup_res = client.post("/api/upload", files=files)
    assert dup_res.status_code == 409

    # 3. View document chunks
    chunks_res = client.get(f"/api/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    assert len(chunks_res.json()["chunks"]) >= 1

    # 4. Ask a question about ID card replacement
    chat_res = client.post("/api/chat", json={"query": "How much does it cost to replace a lost student ID card?"})
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["grounded"] is True
    assert "$25" in chat_data["answer"]
    assert len(chat_data["citations"]) >= 1
    assert chat_data["citations"][0]["filename"] == "id_card_policy.txt"

    # 5. Delete the document
    del_res = client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200

    # 6. Verify chunks were deleted
    chunks_after = client.get(f"/api/documents/{doc_id}/chunks")
    assert chunks_after.status_code == 404

def test_load_samples_endpoint():
    res = client.post("/api/admin/load-samples")
    assert res.status_code == 200
    data = res.json()
    assert "Sample ingestion complete" in data["message"]
    assert len(data["loaded"]) >= 1
