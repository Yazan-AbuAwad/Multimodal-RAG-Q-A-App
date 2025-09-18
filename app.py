import os
import streamlit as st
from typing import List

from config import AppConfig
from ingestion.loaders import prepare_texts
from indexing.chunking import split_texts
from indexing.vectorstore import build_vectorstore, save_vectorstore, load_vectorstore
from retrieval.qa import make_qa_chain

st.set_page_config(page_title="RAG Q&A App", page_icon="🔎", layout="wide")

cfg = AppConfig.from_env()

st.title("🔎 RAG Q&A App (Local)")

# Sidebar – data source & index persistence
with st.sidebar:
    st.markdown("### Data Source")
    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT", "AUDIO"])

    payload = None
    if input_type == "Link":
        n = st.number_input("Number of links", 1, 20, 1)
        urls = [st.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(n)]
        payload = urls
    elif input_type == "Text":
        payload = st.text_area("Enter the text")
    elif input_type == "PDF":
        payload = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    elif input_type == "DOCX":
        payload = st.file_uploader("Upload DOCX", type=["docx", "doc"], accept_multiple_files=True)
    elif input_type == "TXT":
        payload = st.file_uploader("Upload TXT", type=["txt"], accept_multiple_files=True)
    elif input_type == "AUDIO":
        payload = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a", "ogg"], accept_multiple_files=True)
        st.session_state["asr_backend"] = st.radio("ASR backend", ["faster-whisper", "transformers"], horizontal=True)

    st.markdown("---")
    st.markdown("### Index Persistence")
    index_dir = st.text_input("Index dir (local)", value=cfg.index_dir)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save index") and "vectorstore" in st.session_state:
            save_vectorstore(st.session_state["vectorstore"], index_dir)
            st.success(f"Saved index to {index_dir}")
    with c2:
        if st.button("Load index"):
            try:
                st.session_state["vectorstore"] = load_vectorstore(index_dir)
                st.success(f"Loaded index from {index_dir}")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.markdown("---")
    if st.button("Build index"):
        try:
            raw_texts = prepare_texts(input_type, payload, asr_backend=st.session_state.get("asr_backend"))
            chunks = split_texts(raw_texts, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
            if not chunks:
                st.warning("No text found after parsing/splitting.")
            else:
                st.session_state["vectorstore"] = build_vectorstore(chunks, device=cfg.device)
                st.success(f"Indexed {len(chunks)} chunks.")
        except Exception as e:
            st.error(f"Failed to build index: {e}")

# Query UI
if "vectorstore" in st.session_state:
    st.markdown("### Ask a question")
    col_q, col_k = st.columns([3,1])
    with col_q:
        query = st.text_input("Your question", placeholder="Ask about your indexed content…")
    with col_k:
        top_k = st.slider("Top‑k", 2, 10, cfg.top_k)

    if st.button("Submit"):
        try:
            chain = make_qa_chain(st.session_state["vectorstore"], cfg, k=top_k)
            result = chain({"query": query})
            st.markdown("#### Answer")
            st.write(result.get("result", ""))

            srcs = result.get("source_documents", [])
            if srcs:
                with st.expander("Sources"):
                    for i, d in enumerate(srcs, 1):
                        meta = d.metadata or {}
                        src = meta.get("source") or meta.get("url") or meta.get("file") or ""
                        start = meta.get("start")
                        end = meta.get("end")
                        preview = (d.page_content or "").strip().replace("\n", " ")
                        if len(preview) > 300:
                            preview = preview[:300] + "…"
                        tstamp = f" [ {start:.1f}s → {end:.1f}s ]" if start is not None else ""
                        st.markdown(f"**{i}.** {preview}\n\n— _{src}{tstamp}_")
        except Exception as e:
            st.error(f"Query failed: {e}")
else:
    st.info("Load or build an index in the sidebar to start asking questions.")
