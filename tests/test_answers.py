import pytest
from app.answer_generator import generate_grounded_answer, identify_relevant_office
from app.config import CAMPUS_DIRECTORY

def test_grounded_answer_with_citations():
    retrieval_result = {
        "chunks": [
            {
                "chunk_id": 1,
                "doc_id": 10,
                "filename": "academic_calendar.txt",
                "page_number": 1,
                "section_heading": "Add/Drop Deadlines",
                "content": "The deadline to add a new course without instructor permission is September 15. The final drop deadline for a 100% tuition refund is September 22.",
                "similarity_score": 0.84
            }
        ],
        "passed_threshold": True,
        "top_score": 0.84,
        "total_chunks_indexed": 1
    }

    answer_data = generate_grounded_answer("When is the drop deadline with 100% refund?", retrieval_result)
    assert answer_data["grounded"] is True
    assert "[1]" in answer_data["answer"]
    assert "September 22" in answer_data["answer"] or "September 15" in answer_data["answer"]
    assert len(answer_data["citations"]) == 1
    assert answer_data["citations"][0]["filename"] == "academic_calendar.txt"
    assert "September 22" in answer_data["citations"][0]["excerpt"]

def test_fallback_when_not_passed_threshold():
    retrieval_result = {
        "chunks": [],
        "passed_threshold": False,
        "top_score": 0.28,
        "total_chunks_indexed": 10
    }

    answer_data = generate_grounded_answer("How do I fix my personal laptop screen?", retrieval_result)
    assert answer_data["grounded"] is False
    assert len(answer_data["citations"]) == 0
    assert "could not find specific information" in answer_data["answer"].lower()
    assert answer_data["suggested_office"] is not None

def test_office_routing_heuristics():
    assert "Registrar" in identify_relevant_office("How do I withdraw from a course?")
    assert "Financial Aid" in identify_relevant_office("When is the FAFSA deadline and tuition fee due?")
    assert "Health" in identify_relevant_office("Where can I get mental health counseling or flu vaccine?")
    assert "Parking" in identify_relevant_office("How much does a vehicle parking citation cost?")
    assert "Housing" in identify_relevant_office("When are quiet hours in the dorm?")
