from __future__ import annotations
import tempfile
from typing import Iterable


def ensure_iterable(x) -> Iterable:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return x
    return [x]


def upload_to_tempfile(upload) -> str:
    byts = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    suffix = "." + (upload.name.split(".")[-1].lower() if hasattr(upload, "name") else "bin")
    fp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fp.write(byts)
    fp.flush()
    return fp.name
