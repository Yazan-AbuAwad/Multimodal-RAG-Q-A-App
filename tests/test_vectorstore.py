from indexing.vectorstore import build_vectorstore

def test_build_vectorstore():
    vs = build_vectorstore(["hello world", "another doc"], device="cpu")
    assert vs is not None
