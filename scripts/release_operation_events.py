"""Durable sanitized operation timing, separate from release eligibility receipts."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid


def measured(path, category, operation, function, *, clock=time.monotonic):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Operation evidence cannot be a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    identifier = str(uuid.uuid4())
    def append(event, **fields):
        value = dict(schemaVersion=1, id=identifier, category=category, operation=operation,
                     event=event, observedAt=datetime.now(timezone.utc).isoformat(), **fields)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, 'a') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    append('started')
    start = clock()
    try:
        value = function()
    except BaseException:
        # Do not record exception strings: subprocess details may contain private data.
        append('failed', elapsedSeconds=max(0, clock() - start))
        raise
    append('completed', elapsedSeconds=max(0, clock() - start))
    return value
