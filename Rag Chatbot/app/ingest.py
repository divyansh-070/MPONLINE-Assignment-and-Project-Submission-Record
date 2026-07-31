"""
Builds the vector index used for retrieval.

Pipeline:
  1. Fetch the quarterly report (PDF or HTML, from a URL or local file)
  2. Extract plain text
  3. Split into overlapping chunks
  4. Embed each chunk with a sentence-transformers model
  5. Store the vectors in a FAISS index + the raw chunks in a JSON sidecar

Run directly to (re)build the index:
    python -m app.ingest
"""
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app import config


def _looks_like_pdf(content: bytes, url_or_path: str) -> bool:
    if url_or_path.lower().endswith(".pdf"):
        return True
    return content[:5] == b"%PDF-"


def _extract_text_from_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _extract_text_from_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines left behind by table/markup stripping
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text


def fetch_source_text() -> str:
    """Fetch the configured source (local file takes priority over URL) and
    return its plain-text content."""
    if config.LOCAL_SOURCE_PATH:
        path = Path(config.LOCAL_SOURCE_PATH)
        if not path.is_absolute():
            path = config.BASE_DIR / path
        print(f"[ingest] Reading local source: {path}")
        content = path.read_bytes()
        source_ref = str(path)
    else:
        print(f"[ingest] Downloading source: {config.SOURCE_URL}")
        resp = requests.get(
            config.SOURCE_URL,
            headers={"User-Agent": "amazon-rag-chatbot/1.0 (contact: example@example.com)"},
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.content
        source_ref = config.SOURCE_URL

    if _looks_like_pdf(content, source_ref):
        print("[ingest] Detected PDF, extracting text with pypdf...")
        text = _extract_text_from_pdf(content)
    else:
        print("[ingest] Detected HTML, extracting text with BeautifulSoup...")
        text = _extract_text_from_html(content)

    if not text.strip():
        raise RuntimeError("No text could be extracted from the source document.")

    print(f"[ingest] Extracted {len(text):,} characters of text.")
    return text


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Simple, dependency-free sliding-window chunker that prefers to break
    on paragraph/sentence boundaries so chunks stay coherent."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    # Normalize whitespace first
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the tail of the previous one
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n{para}".strip()
            # If a single paragraph is longer than chunk_size, hard-split it
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]
    if current:
        chunks.append(current)

    # Filter out near-empty / boilerplate-only chunks
    chunks = [c for c in chunks if len(c) > 40]
    print(f"[ingest] Split into {len(chunks)} chunks (~{chunk_size} chars each).")
    return chunks


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Embed chunks using the Gemini Embeddings API (no PyTorch required)."""
    from google import genai

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY must be set to build the index.")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    vectors = []
    batch_size = 100  # Gemini supports up to 100 texts per call

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL_NAME,
            contents=batch,
        )
        for emb in response.embeddings:
            vectors.append(emb.values)
        print(f"[ingest] Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
        if i + batch_size < len(chunks):
            import time
            time.sleep(15)  # Strict rate limit evasion

    arr = np.asarray(vectors, dtype="float32")
    # Normalise so inner product == cosine similarity in FAISS IndexFlatIP
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-9)
    return arr



def build_index(force: bool = False) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if config.FAISS_INDEX_PATH.exists() and config.CHUNKS_PATH.exists() and not force:
        print("[ingest] Index already exists, skipping build. Pass force=True to rebuild.")
        return

    import faiss  # local import: keep module import light for callers that just read the index

    text = fetch_source_text()
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("No chunks produced from source text.")

    vectors = embed_chunks(chunks)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=0), encoding='utf-8')
    config.META_PATH.write_text(
        json.dumps(
            {
                "source": config.LOCAL_SOURCE_PATH or config.SOURCE_URL,
                "source_label": config.SOURCE_LABEL,
                "num_chunks": len(chunks),
                "embedding_model": config.EMBEDDING_MODEL_NAME,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f"[ingest] Index built: {len(chunks)} chunks -> {config.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv
    build_index(force=force_rebuild)
