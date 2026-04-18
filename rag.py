"""
RAG pipeline — split into:

1. **Base corpus** (shared, persisted on disk)
   The PDFs shipped with the Space. Built once at startup, reused by every
   session. Uses local sentence-transformers embeddings — no API key needed.

2. **Session corpus** (per-user, in-memory, ephemeral)
   PDFs uploaded through the UI by a single Gradio session. Stored in an
   in-memory Chroma collection with a unique name, GC'd when the session ends.
   NEVER persisted to disk, so no cross-user leakage.

The `query_private_database` tool merges results from both retrievers at
query time — users get their own uploads plus the bundled base corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = APP_DIR / "data" / "Companies-AI-Initiatives"
CHROMA_DIR = APP_DIR / "data" / "chroma_index"
MANIFEST_PATH = CHROMA_DIR / "manifest.json"
BASE_COLLECTION = "AI_Initiatives_base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_ZIP_CANDIDATES = [
    APP_DIR / "data" / "Companies-AI-Initiatives.zip",
    APP_DIR / "data" / "Healthcare-AI-Initiatives.zip",
]

MAX_SESSION_FILES = 10
MAX_SESSION_FILE_BYTES = 25 * 1024 * 1024  # 25 MB

_base_retriever = None
_base_status = "Base corpus not loaded."
_embedding_model = None


# ---------------------------------------------------------------------------
# Shared embedding model — loaded lazily, reused across base + sessions
# ---------------------------------------------------------------------------
def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


# ---------------------------------------------------------------------------
# Base corpus — shared, persisted
# ---------------------------------------------------------------------------
def _ensure_base_pdf_dir() -> Optional[Path]:
    pdf_dir = Path(os.environ.get("PDF_DIR", DEFAULT_PDF_DIR))
    pdf_dir.mkdir(parents=True, exist_ok=True)
    if any(pdf_dir.glob("*.pdf")):
        return pdf_dir
    for zip_path in DEFAULT_ZIP_CANDIDATES:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(pdf_dir)
            if any(pdf_dir.glob("*.pdf")):
                return pdf_dir
    return None


def _corpus_fingerprint(pdf_dir: Path) -> str:
    items = sorted(
        (p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in pdf_dir.glob("*.pdf")
    )
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()[:16]


def _read_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _write_manifest(data: dict) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _try_load_cached_base(pdf_dir: Path):
    manifest = _read_manifest()
    if manifest.get("fingerprint") != _corpus_fingerprint(pdf_dir):
        return None
    if manifest.get("embed_model") != EMBED_MODEL_NAME:
        return None
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        return None
    try:
        vs = Chroma(
            collection_name=BASE_COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=_get_embedding_model(),
        )
        if vs._collection.count() == 0:
            return None
        return vs
    except Exception:
        return None


def build_base_pipeline(force_rebuild: bool = False) -> str:
    """Build or reload the shared base retriever. Safe to call at startup."""
    global _base_retriever, _base_status

    pdf_dir = _ensure_base_pdf_dir()
    if pdf_dir is None:
        _base_retriever = None
        _base_status = (
            "ℹ️ No bundled PDFs — users' uploaded PDFs will be the only RAG source."
        )
        return _base_status

    try:
        if not force_rebuild:
            cached = _try_load_cached_base(pdf_dir)
            if cached is not None:
                _base_retriever = cached.as_retriever(
                    search_type="similarity", search_kwargs={"k": 6}
                )
                n = cached._collection.count()
                _base_status = (
                    f"✅ Base corpus loaded from cache — {n} chunks ({pdf_dir.name}/)."
                )
                return _base_status

        loader = PyPDFDirectoryLoader(path=str(pdf_dir))
        docs = loader.load()
        if not docs:
            _base_retriever = None
            _base_status = f"ℹ️ No readable PDFs in {pdf_dir} — base corpus disabled."
            return _base_status

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        if CHROMA_DIR.exists():
            import shutil

            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=_get_embedding_model(),
            collection_name=BASE_COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )
        _base_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )
        _write_manifest(
            {
                "fingerprint": _corpus_fingerprint(pdf_dir),
                "embed_model": EMBED_MODEL_NAME,
                "chunks": len(chunks),
                "docs": len(docs),
                "pdf_dir": str(pdf_dir),
            }
        )
        _base_status = (
            f"✅ Base corpus built — {len(docs)} PDFs → {len(chunks)} chunks "
            f"({pdf_dir.name}/)."
        )
        return _base_status
    except Exception as e:
        _base_retriever = None
        _base_status = f"⚠️ Base corpus build failed: {e}"
        return _base_status


def get_base_retriever():
    return _base_retriever


def base_status() -> str:
    return _base_status


# ---------------------------------------------------------------------------
# Session corpus — per-user, in-memory, ephemeral
# ---------------------------------------------------------------------------
def build_session_retriever(file_paths: List[str]):
    """
    Build an in-memory retriever over a user's uploaded PDFs.

    Returns (retriever, info_dict). Raises ValueError on bad input.
    The retriever holds an in-memory Chroma collection — no disk persistence,
    no cross-session leakage. GC'd when the caller releases the reference.
    """
    if not file_paths:
        raise ValueError("No files to index.")
    if len(file_paths) > MAX_SESSION_FILES:
        raise ValueError(
            f"Too many files ({len(file_paths)}). Max {MAX_SESSION_FILES} per session."
        )

    all_docs = []
    indexed_names: List[str] = []
    skipped: List[str] = []

    for path in file_paths:
        p = Path(path)
        if not p.exists():
            skipped.append(f"{p.name} (missing)")
            continue
        if p.stat().st_size > MAX_SESSION_FILE_BYTES:
            skipped.append(f"{p.name} (>{MAX_SESSION_FILE_BYTES // (1024*1024)}MB)")
            continue
        if p.suffix.lower() != ".pdf":
            skipped.append(f"{p.name} (not a PDF)")
            continue
        try:
            docs = PyPDFLoader(str(p)).load()
            if not docs:
                skipped.append(f"{p.name} (empty / encrypted)")
                continue
            all_docs.extend(docs)
            indexed_names.append(p.name)
        except Exception as e:
            skipped.append(f"{p.name} ({type(e).__name__})")

    if not all_docs:
        raise ValueError(
            "No readable PDFs in upload. "
            + (f"Skipped: {', '.join(skipped)}" if skipped else "")
        )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)

    # In-memory Chroma (no persist_directory) → ephemeral, GC'd with the session
    collection_name = f"session_{uuid.uuid4().hex[:12]}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embedding_model(),
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    return retriever, {
        "indexed": indexed_names,
        "skipped": skipped,
        "chunks": len(chunks),
        "docs": len(all_docs),
        "collection": collection_name,
    }
