from indexing.chunking import split_texts

def test_split_texts_basic():
    chunks = split_texts(["A" * 2500], chunk_size=1000, chunk_overlap=100)
    assert len(chunks) >= 2
