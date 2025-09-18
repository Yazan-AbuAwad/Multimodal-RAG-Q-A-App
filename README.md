# RAG Q&A App (Streamlit, Local)

A modular Retrieval-Augmented Generation app that ingests Links / PDF / DOCX / TXT / **Audio**, builds a FAISS index with `all-mpnet-base-v2`, and answers questions using a **local LLM** (`microsoft/phi-2` by default).

## Features
- Multi-source ingestion with clean loaders
- Robust chunking (RecursiveCharacterTextSplitter)
- Cosine-ready embeddings (normalized) + FAISS
- Index **save/load** to avoid re-indexing
- Source previews with optional audio timestamps
- Optional local ASR backends: `faster-whisper` (recommended) or `transformers`
- Config via env vars
- **No external API needed** (runs offline once models are downloaded)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Model Downloads
On first run, Hugging Face Transformers will download:
- **`microsoft/phi-2`** (~6 GB) for the LLM
- **`all-mpnet-base-v2`** (~400 MB) for embeddings

These are cached locally in `~/.cache/huggingface/transformers` and reused.

## Config (env vars)
- `HF_REPO_ID` – default `microsoft/phi-2`
- `CHUNK_SIZE` (default 1000), `CHUNK_OVERLAP` (120)
- `INDEX_DIR` (default `./index_store`)
- `TOP_K` (default 4)
- `DEVICE` – `cuda` | `cpu` | `auto`
- `ASR_MODEL` – e.g. `large-v3` (faster-whisper) or `openai/whisper-small` (transformers)

## Tests
```bash
pytest -q
```

## Roadmap
- Optional reranker (e.g., cross-encoder)
- Hybrid retrieval (BM25 + dense)
- Highlighted source spans
- Basic eval notebook in `examples/`

## License
MIT
