import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    # Embeddings / chunking
    chunk_size: int = 1000
    chunk_overlap: int = 120

    # Vectorstore
    index_dir: str = "./index_store"

    # Retrieval defaults
    top_k: int = 4

    # Local LLM
    hf_repo_id: str = "microsoft/phi-2"  # CPU-friendly demo model
    temperature: float = 0.3
    max_new_tokens: int = 256  # smaller for CPU demos
    timeout: int = 60

    # Device hint for embeddings
    device: str = "auto"  # "cuda" | "cpu" | "auto"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 120)),
            index_dir=os.getenv("INDEX_DIR", "./index_store"),
            top_k=int(os.getenv("TOP_K", 4)),
            hf_repo_id=os.getenv("HF_REPO_ID", "microsoft/phi-2"),
            temperature=float(os.getenv("TEMPERATURE", 0.3)),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 256)),
            timeout=int(os.getenv("LLM_TIMEOUT", 60)),
            device=os.getenv("DEVICE", "auto"),
        )
