from __future__ import annotations
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def _device_choice(requested: str) -> str:
    if requested and requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_vectorstore(texts: List[str], *, device: str = "auto") -> FAISS:
    dev = _device_choice(device)
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": dev},
        encode_kwargs={"normalize_embeddings": True},  # cosine ready
    )
    return FAISS.from_texts(texts, embedding=emb)


def save_vectorstore(vs: FAISS, path: str) -> None:
    vs.save_local(path)


def load_vectorstore(path: str) -> FAISS:
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
