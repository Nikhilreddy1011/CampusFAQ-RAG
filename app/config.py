import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"
DB_PATH = DATA_DIR / "campus_faq.db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval & Embedding Configuration
# FastEmbed model: lightweight, runs locally via ONNX without API keys
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Text chunking settings
CHUNK_SIZE = 450          # Target character length per chunk
CHUNK_OVERLAP = 80        # Overlap characters to maintain context across boundaries
MIN_CHUNK_LENGTH = 20     # Discard trivial snippets (e.g. page headers/numbers)

# Retrieval thresholds
TOP_K = 4
SIMILARITY_THRESHOLD = 0.55  # Cosine similarity cutoff (0.0 to 1.0) for relevancy

# Campus fallback contact information
CAMPUS_DIRECTORY = {
    "registrar": "Office of the Registrar (registrar@sampleuniv.edu, 555-0140, Hall 101)",
    "financial_aid": "Financial Aid & Scholarships (finaid@sampleuniv.edu, 555-0141, Hall 108)",
    "housing": "Residence Life & Housing (housing@sampleuniv.edu, 555-0142, Commons B)",
    "health": "Student Health & Wellness Center (healthcenter@sampleuniv.edu, 555-0143, Health Bldg)",
    "parking": "Parking & Transportation Services (parking@sampleuniv.edu, 555-0144, North Garage)",
    "placement": "Career Development & Placement Cell (placements@gitam.edu / Career Guidance Office)",
    "general": "Campus Information & Helpdesk (helpdesk@sampleuniv.edu, 555-0100, Student Union 2nd Fl)"
}

# Privacy & Logging: do not log raw student queries to disk
LOG_RAW_QUERIES = False

# Local LLM Synthesis (Ollama Llama 3.2)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
ENABLE_OLLAMA_SYNTHESIS = True
OLLAMA_TIMEOUT = 50  # seconds
