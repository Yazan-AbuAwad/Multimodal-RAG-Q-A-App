from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_texts(raw_texts: List[str], *, chunk_size: int = 1000, chunk_overlap: int = 120) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks: List[str] = []
    for t in raw_texts:
        if t:
            chunks.extend(splitter.split_text(t))
    return chunks
