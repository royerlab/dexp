import uuid
from pathlib import Path

import fasteners
from arbol import aprint


def create_lock(prefix: str = "") -> fasteners.InterProcessLock:
    """Creates an multi-processing safe lock using unique unit idenfier."""
    identifier = uuid.uuid4().hex
    worker_directory = Path.home() / f".multiprocessing/{identifier}"
    worker_directory.mkdir(exist_ok=True, parents=True)
    aprint(f"Created lock directory {worker_directory}")

    return fasteners.InterProcessLock(path=worker_directory / "lock.file")
