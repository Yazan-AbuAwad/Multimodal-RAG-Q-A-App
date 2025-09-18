from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from config import AppConfig

_model_cache = {}

def get_llm(cfg: AppConfig):
    global _model_cache
    if "llm" in _model_cache:
        return _model_cache["llm"]

    model_id = cfg.hf_repo_id or "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    _model_cache["llm"] = llm
    return llm
