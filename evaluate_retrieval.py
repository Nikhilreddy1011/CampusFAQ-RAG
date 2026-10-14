#!/usr/bin/env python3
"""
Campus FAQ System — Semantic Retrieval & Fallback Evaluation Script
Evaluates retrieval accuracy, Mean Reciprocal Rank (MRR), Hit Rate @ Top-K,
and fallback behavior on realistic campus questions.
"""

import time
from typing import List, Dict, Any
from pathlib import Path

from app.config import SAMPLE_DOCS_DIR, SIMILARITY_THRESHOLD, TOP_K
from app.database import (
    init_db, get_document_by_hash, insert_document_and_chunks,
    get_stats
)
from app.document_processor import process_document, compute_sha256
from app.embeddings import embedding_service
from app.retriever import retrieve_relevant_chunks
from app.answer_generator import generate_grounded_answer

# Ground truth test dataset
EVALUATION_DATASET = [
    # --- Category: Academic & Registrar ---
    {
        "query": "When is the deadline to drop a course for a 100% tuition refund?",
        "expected_doc": "academic_calendar_and_registration.txt",
        "expected_answer_keywords": ["September 22", "100%", "refund"],
        "is_in_domain": True
    },
    {
        "query": "Can an undergraduate student take 20 credits in one semester?",
        "expected_doc": "academic_calendar_and_registration.txt",
        "expected_answer_keywords": ["overload", "3.3", "21 credits"],
        "is_in_domain": True
    },
    {
        "query": "What is the deadline to apply for Spring graduation?",
        "expected_doc": "academic_calendar_and_registration.txt",
        "expected_answer_keywords": ["February 1", "graduation"],
        "is_in_domain": True
    },

    # --- Category: Parking & Transportation ---
    {
        "query": "How much does a commuter parking permit cost per semester?",
        "expected_doc": "campus_parking_and_transportation.txt",
        "expected_answer_keywords": ["$175", "Yellow Lot", "commuter"],
        "is_in_domain": True
    },
    {
        "query": "How many days do I have to appeal a campus parking citation?",
        "expected_doc": "campus_parking_and_transportation.txt",
        "expected_answer_keywords": ["14 calendar days", "appeal"],
        "is_in_domain": True
    },

    # --- Category: Student Health & Counseling ---
    {
        "query": "What are the walk-in hours for acute illness at the health clinic?",
        "expected_doc": "student_health_and_wellness.txt",
        "expected_answer_keywords": ["8:30 AM", "11:30 AM", "walk-in"],
        "is_in_domain": True
    },
    {
        "query": "How many free counseling therapy sessions are students allowed?",
        "expected_doc": "student_health_and_wellness.txt",
        "expected_answer_keywords": ["12", "therapy", "free"],
        "is_in_domain": True
    },

    # --- Category: Residence Life & Dining ---
    {
        "query": "What is the deadline to change or cancel my meal plan?",
        "expected_doc": "housing_and_dining_policies.docx",
        "expected_answer_keywords": ["second Friday", "September 8", "meal plan"],
        "is_in_domain": True
    },
    {
        "query": "When do quiet hours begin in the residence halls on weeknights?",
        "expected_doc": "housing_and_dining_policies.docx",
        "expected_answer_keywords": ["10:00 PM", "8:00 AM", "Quiet hours"],
        "is_in_domain": True
    },

    # --- Category: Financial Aid & Student Accounts ---
    {
        "query": "What is the priority FAFSA filing deadline for need-based grants?",
        "expected_doc": "student_financial_services.pdf",
        "expected_answer_keywords": ["February 15", "FAFSA", "priority"],
        "is_in_domain": True
    },
    {
        "query": "What is the starting hourly pay rate for Federal Work-Study?",
        "expected_doc": "student_financial_services.pdf",
        "expected_answer_keywords": ["$15.50", "hourly", "work-study"],
        "is_in_domain": True
    },
    {
        "query": "What minimum GPA is required for Satisfactory Academic Progress?",
        "expected_doc": "student_financial_services.pdf",
        "expected_answer_keywords": ["2.0", "cumulative GPA"],
        "is_in_domain": True
    },

    # --- Category: Out-of-Domain / Negative Controls (Must Trigger Fallback) ---
    {
        "query": "What is the recipe for baking chocolate chip cookies with sea salt?",
        "expected_doc": None,
        "expected_answer_keywords": [],
        "is_in_domain": False
    },
    {
        "query": "How do I replace the alternator on a 2012 Honda Civic sedan?",
        "expected_doc": None,
        "expected_answer_keywords": [],
        "is_in_domain": False
    },
    {
        "query": "What is the capital city of Australia and its current population?",
        "expected_doc": None,
        "expected_answer_keywords": [],
        "is_in_domain": False
    }
]

def ensure_sample_documents_loaded():
    """Ensure all sample documents are indexed before running evaluation."""
    init_db()
    sample_files = list(SAMPLE_DOCS_DIR.glob("*.*"))
    for p in sample_files:
        if p.suffix.lower() not in [".pdf", ".docx", ".txt"]:
            continue
        content_bytes = p.read_bytes()
        file_hash = compute_sha256(content_bytes)
        if not get_document_by_hash(file_hash):
            chunks, file_type = process_document(p.name, content_bytes)
            chunk_texts = [c["content"] for c in chunks]
            embs = embedding_service.embed_texts(chunk_texts)
            insert_document_and_chunks(
                filename=p.name,
                file_type=file_type,
                file_size=len(content_bytes),
                sha256_hash=file_hash,
                chunks=chunks,
                embeddings=embs
            )

def run_evaluation():
    print("=" * 80)
    print("Campus FAQ System — Retrieval & Grounding Evaluation")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD} | Top-K: {TOP_K}")
    print("=" * 80)

    ensure_sample_documents_loaded()
    stats = get_stats()
    print(f"Database status: {stats['total_documents']} documents, {stats['total_chunks']} chunks indexed.\n")

    in_domain_total = 0
    in_domain_hit1 = 0
    in_domain_hit3 = 0
    reciprocal_ranks = []
    latencies = []

    out_domain_total = 0
    out_domain_fallback_correct = 0

    results_table = []

    for item in EVALUATION_DATASET:
        query = item["query"]
        expected = item["expected_doc"]
        is_in_domain = item["is_in_domain"]

        t0 = time.perf_counter()
        retrieval = retrieve_relevant_chunks(query, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD)
        answer = generate_grounded_answer(query, retrieval)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_chunks = retrieval["chunks"]
        top_score = retrieval["top_score"]
        passed = retrieval["passed_threshold"]

        if is_in_domain:
            in_domain_total += 1
            retrieved_docs = [c["filename"] for c in retrieved_chunks]

            # Check Hit@1
            hit1 = len(retrieved_docs) > 0 and retrieved_docs[0] == expected
            if hit1:
                in_domain_hit1 += 1

            # Check Hit@3
            hit3 = expected in retrieved_docs[:3]
            if hit3:
                in_domain_hit3 += 1

            # Reciprocal rank
            rr = 0.0
            for rank_idx, doc_name in enumerate(retrieved_docs):
                if doc_name == expected:
                    rr = 1.0 / (rank_idx + 1)
                    break
            reciprocal_ranks.append(rr)

            status = "HIT" if hit3 else "MISS"
            top_retrieved = retrieved_docs[0] if retrieved_docs else "None"
            results_table.append({
                "query": query[:42] + ("..." if len(query) > 42 else ""),
                "expected": expected[:22],
                "retrieved": top_retrieved[:22],
                "score": f"{top_score:.3f}",
                "status": status,
                "latency": f"{latency_ms:.1f}ms"
            })
        else:
            out_domain_total += 1
            # Out of domain query should NOT pass threshold
            correct_fallback = (not passed) and (not answer["grounded"])
            if correct_fallback:
                out_domain_fallback_correct += 1

            status = "FALLBACK (CORRECT)" if correct_fallback else "FALSE_POSITIVE"
            top_retrieved = retrieved_chunks[0]["filename"] if retrieved_chunks else "None"
            results_table.append({
                "query": query[:42] + ("..." if len(query) > 42 else ""),
                "expected": "FALLBACK",
                "retrieved": f"Score: {top_score:.3f}",
                "score": f"{top_score:.3f}",
                "status": status,
                "latency": f"{latency_ms:.1f}ms"
            })

    # Print results table
    print(f"{'Question':<45} {'Expected Doc':<24} {'Top Result':<24} {'Score':<8} {'Status':<18} {'Latency'}")
    print("-" * 130)
    for r in results_table:
        print(f"{r['query']:<45} {r['expected']:<24} {r['retrieved']:<24} {r['score']:<8} {r['status']:<18} {r['latency']}")
    print("-" * 130)

    # Compute final metrics
    hit_rate_1 = (in_domain_hit1 / in_domain_total) * 100 if in_domain_total else 0.0
    hit_rate_3 = (in_domain_hit3 / in_domain_total) * 100 if in_domain_total else 0.0
    mrr = (sum(reciprocal_ranks) / in_domain_total) if in_domain_total else 0.0
    fallback_rate = (out_domain_fallback_correct / out_domain_total) * 100 if out_domain_total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "=" * 50)
    print("SUMMARY BENCHMARK METRICS (MEASURED)")
    print("=" * 50)
    print(f"Total Test Questions:            {len(EVALUATION_DATASET)}")
    print(f"In-Domain Questions:             {in_domain_total}")
    print(f"Out-of-Domain / Negative:        {out_domain_total}")
    print(f"In-Domain Hit Rate @ Top-1:      {hit_rate_1:.1f}% ({in_domain_hit1}/{in_domain_total})")
    print(f"In-Domain Hit Rate @ Top-3:      {hit_rate_3:.1f}% ({in_domain_hit3}/{in_domain_total})")
    print(f"Mean Reciprocal Rank (MRR):      {mrr:.3f}")
    print(f"Out-of-Domain Fallback Rate:     {fallback_rate:.1f}% ({out_domain_fallback_correct}/{out_domain_total})")
    print(f"Average End-to-End Latency:      {avg_latency:.2f} ms")
    print("=" * 50)

if __name__ == "__main__":
    run_evaluation()
