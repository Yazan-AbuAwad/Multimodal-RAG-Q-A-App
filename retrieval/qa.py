from __future__ import annotations
from langchain.chains import RetrievalQA
from config import AppConfig
from .llm import get_llm


def make_qa_chain(vs, cfg: AppConfig, *, k: int | None = None):
    retriever = vs.as_retriever(search_kwargs={"k": k or cfg.top_k})
    llm = get_llm(cfg)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
    return chain
