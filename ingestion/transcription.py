from __future__ import annotations
import os
from typing import Optional

from utils.io import upload_to_tempfile


def transcribe_audio(upload, backend: str = "faster-whisper", language: Optional[str] = None) -> str:
    path = upload_to_tempfile(upload)
    if backend == "faster-whisper":
        return _asr_faster_whisper(path, language)
    elif backend == "transformers":
        return _asr_transformers(path)
    else:
        raise ValueError(f"Unknown ASR backend: {backend}")


def _asr_faster_whisper(path: str, language: Optional[str]) -> str:
    from faster_whisper import WhisperModel
    device, compute_type = _device()
    model_name = os.getenv("ASR_MODEL", "large-v3")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, _ = model.transcribe(
        path,
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def _asr_transformers(path: str) -> str:
    from transformers import pipeline
    model_id = os.getenv("ASR_MODEL", "openai/whisper-small")
    asr = pipeline("automatic-speech-recognition", model=model_id)
    result = asr(path)
    return (result[0]["text"] if isinstance(result, list) else result["text"]).strip()


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"
