import json
import logging
import re
import urllib.request
from typing import List, Dict, Any, Optional

from app.config import (
    CAMPUS_DIRECTORY, ENABLE_OLLAMA_SYNTHESIS,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
)

logger = logging.getLogger("campus_faq.answer_generator")

GENERIC_STOPWORDS = {
    "how", "was", "were", "is", "are", "the", "in", "at", "for", "of", "and", "or", "to", "a", "an",
    "gitam", "university", "campus", "college", "school", "student", "students", "tell", "me", "about",
    "what", "when", "where", "which", "who", "why", "does", "do", "did", "can", "could", "would", "should"
}

def identify_relevant_office(query: str) -> str:
    """Heuristic keyword matching to route student questions to the appropriate campus office."""
    q = query.lower()
    if any(k in q for k in ["placement", "placements", "salary", "package", "recruiter", "company", "companies", "job", "career", "hiring", "internship"]):
        return CAMPUS_DIRECTORY["placement"]
    elif any(k in q for k in ["fee", "tuition", "finaid", "financial aid", "scholarship", "fafsa", "grant", "work study", "payment plan"]):
        return CAMPUS_DIRECTORY["financial_aid"]
    elif any(k in q for k in ["drop", "add", "withdraw", "course", "grade", "transcript", "credit", "prerequisite", "register", "registration", "degree", "diploma", "calendar"]):
        return CAMPUS_DIRECTORY["registrar"]
    elif any(k in q for k in ["dorm", "housing", "room", "roommate", "dining", "meal", "cafeteria", "residence", "move-in", "move-out"]):
        return CAMPUS_DIRECTORY["housing"]
    elif any(k in q for k in ["health", "doctor", "clinic", "counseling", "mental", "therapy", "sick", "vaccine", "immunization", "nurse", "prescription"]):
        return CAMPUS_DIRECTORY["health"]
    elif any(k in q for k in ["parking", "permit", "car", "vehicle", "ticket", "citation", "shuttle", "bus", "bike"]):
        return CAMPUS_DIRECTORY["parking"]
    return CAMPUS_DIRECTORY["general"]

def score_sentence_relevance(sentence: str, query_words: set) -> float:
    """Score a sentence by lexical overlap with user query, favoring informative content."""
    sent_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', sentence.lower())) - GENERIC_STOPWORDS
    if not sent_words:
        return 0.0
    common = sent_words.intersection(query_words)
    overlap_ratio = len(common) / (len(query_words) + 1e-5)
    # Give slight bonus to sentences containing actionable keywords (dates, numbers, policies)
    bonus = 0.0
    if re.search(r'\b(must|deadline|fee|require|submit|contact|within|before|after|hours?|online|portal|\$|\d+)\b', sentence.lower()):
        bonus = 0.15
    return overlap_ratio + bonus

def synthesize_with_ollama(query: str, citations: List[Dict[str, Any]]) -> Optional[str]:
    """
    Generate a natural, strictly grounded answer using local Ollama (Llama 3.2).
    Optimized for fast CPU inference and clean inline citations without raw filenames.
    """
    if not ENABLE_OLLAMA_SYNTHESIS or not citations:
        return None

    # Use top 3 most relevant citations to minimize prompt eval latency on CPU
    top_citations = citations[:3]

    # Provide concise excerpts with citation IDs only, omitting raw file paths
    context_blocks = []
    for c in top_citations:
        context_blocks.append(f"[{c['id']}] {c['excerpt']}")
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an official Campus FAQ Helpdesk assistant.\n"
        "Answer the student's question clearly, politely, and concisely using ONLY the provided facts.\n\n"
        "STRICT GROUNDING & CITATION RULES:\n"
        "1. Rely strictly on facts in the excerpts. Do NOT assume, fabricate, or extrapolate any fees, dates, or policies.\n"
        "2. If the excerpts do not answer the question or lack information on the specific topic, clearly state that the information is not available in the campus documents and suggest contacting the appropriate campus office.\n"
        "3. Cite facts using ONLY clean inline brackets like [1] or [2] (e.g., 'Tuition fees depend on the programme and campus [1].').\n"
        "4. Never write out 'Source:' or file names inside your sentences.\n"
        "5. Keep the response direct and concise (1-2 short paragraphs)."
    )

    user_prompt = f"Official Context Excerpts:\n{context_text}\n\nStudent Question: {query}\n\nAnswer:"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 140,    # Optimal length for fast CPU generation
            "num_ctx": 2048,        # Low memory footprint for snappy prompt evaluation
            "top_p": 0.9
        }
    }

    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer_content = data.get("message", {}).get("content", "").strip()
            if answer_content:
                # Post-process cleanup in case model outputs redundant 'Source:' strings
                answer_content = re.sub(r'\[(\d+)\]\s*Source:[^,\n]+(?:,\s*which\s+states\s+that)?', r'[\1]', answer_content, flags=re.IGNORECASE)
                answer_content = re.sub(r'Source:\s*[^,\n]+\s*\[(\d+)\]', r'[\1]', answer_content, flags=re.IGNORECASE)
                logger.info("Successfully synthesized grounded answer via Ollama (%s)", OLLAMA_MODEL)
                return answer_content
    except Exception as e:
        logger.info("Ollama synthesis skipped or unavailable (%s). Falling back to extractive mode.", e)

    return None

def generate_grounded_answer(
    query: str,
    retrieval_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Produce a concise, reliable answer grounded in retrieved document chunks.
    Attempts local LLM synthesis via Ollama (Llama 3.2); if Ollama is unreachable,
    smoothly falls back to verified extractive synthesis.
    """
    chunks = retrieval_result.get("chunks", [])
    passed_threshold = retrieval_result.get("passed_threshold", False)
    top_score = retrieval_result.get("top_score", 0.0)

    # Extract non-generic topic keywords from query
    query_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())]
    topic_keywords = [w for w in query_tokens if w not in GENERIC_STOPWORDS]

    # Verify topical overlap: do any of the retrieved chunks contain the specific topic keyword?
    combined_chunk_text = " ".join([c.get("content", "").lower() for c in chunks])
    has_topic_match = True
    if topic_keywords and combined_chunk_text:
        has_topic_match = any(
            (kw in combined_chunk_text or (len(kw) > 4 and kw[:4] in combined_chunk_text))
            for kw in topic_keywords
        )

    # Fallback when documents do not support an answer or do not match the key topic
    if not passed_threshold or not chunks or not has_topic_match:
        suggested_office = identify_relevant_office(query)
        topic_phrase = f"regarding '{' '.join(topic_keywords)}'" if topic_keywords else "regarding this topic"
        fallback_text = (
            f"I could not find specific information {topic_phrase} in the uploaded campus documents.\n\n"
            "To get accurate guidance, please reach out to the appropriate campus department:\n"
            f"• **{suggested_office}**\n\n"
            "You may also check with the Campus Helpdesk at helpdesk@sampleuniv.edu or visit the Student Services portal."
        )
        return {
            "answer": fallback_text,
            "grounded": False,
            "confidence_score": top_score,
            "citations": [],
            "suggested_office": suggested_office,
            "synthesizer": "fallback"
        }

    # Extract keywords from query
    query_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))

    # Build citations list
    citations = []
    for idx, chunk in enumerate(chunks):
        citation_id = idx + 1
        filename = chunk["filename"]
        page = chunk.get("page_number", 1)
        heading = chunk.get("section_heading", "General")
        score = chunk.get("similarity_score", 0.0)
        content = chunk["content"].strip()

        source_label = f"{filename} (Page {page})" if page and page > 1 else f"{filename} — {heading}"

        citations.append({
            "id": citation_id,
            "source_label": source_label,
            "filename": filename,
            "page_number": page,
            "section_heading": heading,
            "similarity_score": score,
            "similarity_percent": int(round(score * 100)),
            "excerpt": content
        })

    # 1. Attempt LLM synthesis via Ollama (Llama 3.2)
    llm_answer = synthesize_with_ollama(query, citations)
    if llm_answer:
        return {
            "answer": llm_answer,
            "grounded": True,
            "confidence_score": top_score,
            "citations": citations,
            "suggested_office": None,
            "synthesizer": "llama3.2"
        }

    # 2. Extractive Mode Fallback: filter out raw question headers and metadata lines
    extracted_points = []
    seen_excerpts = set()

    for citation in citations:
        citation_id = citation["id"]
        content = citation["excerpt"]
        raw_lines = [l.strip() for l in content.split("\n") if l.strip()]

        for line in raw_lines:
            # Skip metadata lines or FAQ question headers in bullet lists
            if line.lower().startswith("source:") or line.lower().startswith("http") or len(line) < 15:
                continue
            if line.endswith("?") and not any(kw in line.lower() for kw in ["must", "fee", "cost", "$", "deadline"]):
                continue

            sentences = re.split(r'(?<=[.!?])\s+', line)
            for s in sentences:
                clean_s = s.strip()
                if len(clean_s) < 20:
                    continue
                norm_key = re.sub(r'\s+', ' ', clean_s.lower())
                if norm_key not in seen_excerpts:
                    seen_excerpts.add(norm_key)
                    s_score = score_sentence_relevance(clean_s, query_words)
                    extracted_points.append({
                        "text": clean_s,
                        "citation_id": citation_id,
                        "score": s_score
                    })

    extracted_points.sort(key=lambda x: x["score"], reverse=True)

    if extracted_points:
        primary_point = extracted_points[0]
        primary_sentence = primary_point["text"]
        if not primary_sentence.endswith((".", "!", "?")):
            primary_sentence += "."
        primary_sentence += f" [{primary_point['citation_id']}]"

        answer_lines = [primary_sentence]
        remaining_points = [p for p in extracted_points[1:4] if p["score"] > 0]
        if remaining_points:
            answer_lines.append("\n**Key details from campus documents:**")
            for pt in remaining_points:
                s = pt["text"]
                if not s.endswith((".", "!", "?")):
                    s += "."
                answer_lines.append(f"• {s} [{pt['citation_id']}]")

        answer_text = "\n".join(answer_lines)
    else:
        top_chunk = chunks[0]
        answer_text = f"{top_chunk['content']} [1]"

    return {
        "answer": answer_text,
        "grounded": True,
        "confidence_score": top_score,
        "citations": citations,
        "suggested_office": None,
        "synthesizer": "extractive"
    }
