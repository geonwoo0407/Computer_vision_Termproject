"""Create one JSONL file per vision test without losing previous runs."""

from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


def open_run_log(mode, requested=None):
    if requested is None:
        folder = Path("output/logs")
        stem, suffix = mode, ".jsonl"
    else:
        path = Path(requested)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = path.open("x", encoding="utf-8")
        except FileExistsError:
            if not path.is_file():
                raise
            folder, stem, suffix = path.parent, path.stem, path.suffix
        else:
            print(f"Test log: {path.resolve()}", flush=True)
            return stream

    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder,
                                prefix=f"{stem}_{stamp}_", suffix=suffix, delete=False)
    print(f"Test log: {Path(stream.name).resolve()}", flush=True)
    return stream
