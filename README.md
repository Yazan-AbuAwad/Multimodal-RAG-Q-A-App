# Multimodal RAG Q&A App

An advanced **Retrieval-Augmented Generation (RAG)** question-answering system built with **Streamlit**, **FAISS**, and **Hugging Face**. This app processes various document types; including PDF, DOCX, TXT, and web links, creates vector embeddings, and implements a powerful Q&A interface.


## Features

- Process multiple document types: PDF, DOCX, TXT, and URLs (audio files coming soon!)
- Create vector embeddings for documents using Hugging Face models
- Efficient similarity search with FAISS indexing
- Interactive user interface powered by Streamlit
- Retrieval-Augmented Generation for accurate and context-aware answers
- **Voice input support** (coming soon) for spoken questions

## How It Works

1. **Document Ingestion:** Upload or provide documents and web links.
2. **Embedding Generation:** Convert the document content into vector embeddings.
3. **Indexing:** Store embeddings in a FAISS index for fast retrieval.
4. **Query Processing:** Retrieve relevant documents and generate answers using Hugging Face language models.

## In development:
1. **Transcription (for audio):** Convert voice documents to text using speech-to-text models.
2. **Voice Input:** (In development) Speak your questions instead of typing.

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/multimodal-rag-qa-app.git
   cd multimodal-rag-qa-app
   ```


2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```

3. Run the app
   ```bash
   streamlit run app.py
   ```
