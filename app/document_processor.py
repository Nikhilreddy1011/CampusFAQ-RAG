import hashlib
import io
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pypdf
import docx
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH

def compute_sha256(content: bytes) -> str:
    """Calculate SHA-256 hash of file bytes for deduplication."""
    return hashlib.sha256(content).hexdigest()

def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable control characters."""
    text = re.sub(r'[\r\f\v]+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundary matching."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text_blocks(
    blocks: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Split text blocks (with page and section metadata) into sensible chunks.
    Ensures sentences are not abruptly severed mid-word.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    for block in blocks:
        text = block["text"]
        page_num = block.get("page_number", 1)
        heading = block.get("section_heading", "General")

        if len(text) <= chunk_size:
            if len(text) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "content": text,
                    "page_number": page_num,
                    "section_heading": heading
                })
                chunk_idx += 1
            continue

        sentences = split_into_sentences(text)
        current_chunk = ""
        
        for sentence in sentences:
            if not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += " " + sentence
            else:
                if len(current_chunk) >= MIN_CHUNK_LENGTH:
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "content": current_chunk,
                        "page_number": page_num,
                        "section_heading": heading
                    })
                    chunk_idx += 1
                
                # Overlap: preserve end of previous chunk if long enough
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    overlap_seed = current_chunk[-chunk_overlap:]
                    # align to last word boundary
                    if " " in overlap_seed:
                        overlap_seed = overlap_seed.split(" ", 1)[-1]
                    current_chunk = overlap_seed + " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk and len(current_chunk) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "chunk_index": chunk_idx,
                "content": current_chunk,
                "page_number": page_num,
                "section_heading": heading
            })
            chunk_idx += 1

    return chunks

def extract_from_txt(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract structured text blocks from TXT file with heading detection."""
    try:
        raw_text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = file_bytes.decode("latin-1", errors="replace")

    raw_text = clean_text(raw_text)
    paragraphs = raw_text.split("\n\n")

    blocks = []
    current_heading = "General"

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue

        # Check if this paragraph is a section header (e.g. "## Heading" or "SECTION 1: ..." or short all-caps)
        lines = p.split("\n")
        first_line = lines[0].strip()
        if (first_line.startswith("#") or 
            first_line.endswith(":") or 
            (len(first_line) < 60 and first_line.isupper() and len(lines) == 1)):
            current_heading = first_line.lstrip("#").strip(":").strip()
            # If paragraph contains more content after heading
            if len(lines) > 1:
                content = "\n".join(lines[1:]).strip()
                if content:
                    blocks.append({
                        "text": content,
                        "page_number": 1,
                        "section_heading": current_heading
                    })
            continue

        blocks.append({
            "text": p,
            "page_number": 1,
            "section_heading": current_heading
        })

    return blocks

import fitz

_ocr_engine = None

def get_ocr_engine():
    """Lazy loader for RapidOCR engine to process scanned or image-based PDFs."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception as e:
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None

def extract_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract structured text from PDF per page using PyMuPDF (fitz) and pypdf.
    If the PDF is image-based or scanned (yielding no text layer),
    automatically falls back to RapidOCR.
    """
    blocks = []
    total_chars = 0

    # 1. First attempt: standard digital text extraction with PyMuPDF
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for idx, page in enumerate(doc):
            page_num = idx + 1
            raw_text = page.get_text() or ""
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            total_chars += len(cleaned)
            lines = cleaned.split("\n")
            first_line = lines[0].strip()
            heading = f"Page {page_num}"
            if len(first_line) < 80 and not first_line.isdigit():
                heading = first_line.lstrip("#").strip()

            paras = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            if not paras:
                paras = [cleaned]

            for p in paras:
                if len(p) >= MIN_CHUNK_LENGTH:
                    blocks.append({
                        "text": p,
                        "page_number": page_num,
                        "section_heading": heading
                    })
    except Exception:
        # Fallback to pypdf if fitz had an unexpected parse error
        blocks = []
        total_chars = 0
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                page_text = page.extract_text() or ""
                cleaned = clean_text(page_text)
                if not cleaned:
                    continue
                total_chars += len(cleaned)
                paras = [p.strip() for p in cleaned.split("\n\n") if p.strip()] or [cleaned]
                for p in paras:
                    if len(p) >= MIN_CHUNK_LENGTH:
                        blocks.append({
                            "text": p,
                            "page_number": page_num,
                            "section_heading": f"Page {page_num}"
                        })
        except Exception:
            pass

    # 2. If digital text extraction found insufficient text (< 40 chars),
    # this is a scanned, canvas-rendered, or image-based PDF. Run OCR fallback.
    if total_chars < 40:
        ocr = get_ocr_engine()
        if ocr:
            blocks = []
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for idx, page in enumerate(doc):
                    page_num = idx + 1
                    pix = page.get_pixmap(dpi=150)
                    ocr_res, _ = ocr(pix.tobytes("png"))
                    if not ocr_res:
                        continue

                    lines = [line[1].strip() for line in ocr_res if line and len(line) > 1 and line[1].strip()]
                    if not lines:
                        continue

                    # Filter out browser header if present (e.g., date stamps)
                    body_lines = []
                    heading = f"Page {page_num}"
                    for l_idx, line in enumerate(lines):
                        if l_idx == 0 and ("http" in line.lower() or "/202" in line or "/26" in line):
                            continue
                        if heading == f"Page {page_num}" and len(line) < 70 and not line.isdigit():
                            heading = line
                        body_lines.append(line)

                    if body_lines:
                        page_content = "\n".join(body_lines)
                        blocks.append({
                            "text": page_content,
                            "page_number": page_num,
                            "section_heading": heading
                        })
            except Exception:
                pass

    return blocks

def extract_from_docx(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract structured text from DOCX detecting heading styles and tables."""
    doc = docx.Document(io.BytesIO(file_bytes))
    blocks = []
    current_heading = "General"

    # Extract paragraphs
    for p in doc.paragraphs:
        text = clean_text(p.text)
        if not text:
            continue

        # Detect heading styles
        style_name = p.style.name.lower() if p.style else ""
        if "heading" in style_name or "title" in style_name:
            current_heading = text
            continue

        blocks.append({
            "text": text,
            "page_number": 1,
            "section_heading": current_heading
        })

    # Extract tables if present
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            cells = [clean_text(cell.text) for cell in row.cells]
            table_rows.append(" | ".join(c for c in cells if c))
        table_text = "\n".join(table_rows)
        if len(table_text) >= MIN_CHUNK_LENGTH:
            blocks.append({
                "text": table_text,
                "page_number": 1,
                "section_heading": f"{current_heading} (Table)"
            })

    return blocks

def process_document(filename: str, file_bytes: bytes) -> Tuple[List[Dict[str, Any]], str]:
    """
    Dispatcher to process an uploaded document according to its file extension.
    Returns (chunks, file_type).
    Raises ValueError on unsupported or unparseable files.
    """
    ext = Path(filename).suffix.lower()
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    if ext == ".txt":
        file_type = "TXT"
        blocks = extract_from_txt(file_bytes)
    elif ext == ".pdf":
        file_type = "PDF"
        blocks = extract_from_pdf(file_bytes)
    elif ext == ".docx":
        file_type = "DOCX"
        blocks = extract_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .docx, .txt")

    if not blocks:
        raise ValueError(f"No readable text could be extracted from '{filename}'.")

    chunks = chunk_text_blocks(blocks)
    if not chunks:
        raise ValueError(f"Document contains insufficient extractable content after filtering.")

    return chunks, file_type
