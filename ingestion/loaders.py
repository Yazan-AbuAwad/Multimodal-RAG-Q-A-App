from __future__ import annotations
from io import BytesIO
from typing import List, Optional

from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from docx import Document

from utils.io import ensure_iterable
from ingestion.transcription import transcribe_audio


def _read_pdf_to_text(upload) -> str:
    byts = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    reader = PdfReader(BytesIO(byts))
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    return "\n".join(pages).strip()


def _read_docx_to_text(upload) -> str:
    byts = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    doc = Document(BytesIO(byts))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def _read_txt_to_text(upload) -> str:
    byts = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    return byts.decode("utf-8", errors="ignore").strip()


def _load_from_links(urls: List[str]) -> List[str]:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return [d.page_content for d in docs if d.page_content]


def prepare_texts(input_type: str, input_data, *, asr_backend: Optional[str] = None) -> List[str]:
    """Return a list of raw texts from various inputs.
    input_data may be a string, UploadedFile, or list of those.
    """
    if input_type == "Link":
        urls = [u.strip() for u in ensure_iterable(input_data) if isinstance(u, str) and u.strip()]
        if not urls:
            raise ValueError("Please provide at least one valid URL.")
        return _load_from_links(urls)

    if input_type == "Text":
        if not isinstance(input_data, str) or not input_data.strip():
            raise ValueError("Please enter some text.")
        return [input_data]

    if input_type == "PDF":
        files = list(ensure_iterable(input_data))
        if not files:
            raise ValueError("Please upload at least one PDF.")
        return [_read_pdf_to_text(f) for f in files]

    if input_type == "DOCX":
        files = list(ensure_iterable(input_data))
        if not files:
            raise ValueError("Please upload at least one DOCX.")
        return [_read_docx_to_text(f) for f in files]

    if input_type == "TXT":
        files = list(ensure_iterable(input_data))
        if not files:
            raise ValueError("Please upload at least one TXT.")
        return [_read_txt_to_text(f) for f in files]

    if input_type == "AUDIO":
        files = list(ensure_iterable(input_data))
        if not files:
            raise ValueError("Please upload at least one audio file.")
        return [transcribe_audio(f, backend=asr_backend or "faster-whisper") for f in files]

    raise ValueError("Unsupported input type")
