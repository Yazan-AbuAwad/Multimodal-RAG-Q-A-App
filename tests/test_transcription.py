import io
import pytest
from ingestion.transcription import transcribe_audio

class DummyUpload:
    def __init__(self):
        self.name = "fake.wav"
    def getvalue(self):
        return b"not real audio"

def test_transcribe_backend_switch():
    up = DummyUpload()
    with pytest.raises(Exception):
        transcribe_audio(up, backend="faster-whisper")
