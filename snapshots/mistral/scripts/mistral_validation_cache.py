"""Preserve numerical validation attempts and invalidate results after code changes."""
import hashlib
import json
from pathlib import Path
import time


def fingerprint(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def reuse_or_archive(path, current):
    path = Path(path)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get('passed') is True and record.get('fingerprint') == current:
        return True
    # Keep failures and obsolete passes as evidence. Never turn either into a pass.
    history = path.parent / 'validation-history'
    history.mkdir(exist_ok=True)
    archive = history / f'{path.stem}-{time.time_ns()}.json'
    with archive.open('xb') as output:
        output.write(path.read_bytes())
    path.unlink()
    return False
