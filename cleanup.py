"""
cleanup.py
==========
Background task that runs every 10 minutes and deletes
any processed files older than 1 hour from /tmp/shaddies.
"""

import asyncio
import time
from pathlib import Path


async def schedule_cleanup(tmp_dir: Path, interval_seconds: int = 600):
    """
    Runs every `interval_seconds` (default 10 min).
    Deletes files older than 1 hour.
    """
    while True:
        try:
            now = time.time()
            one_hour = 3600

            for f in tmp_dir.glob("*.docx"):
                age = now - f.stat().st_mtime
                if age > one_hour:
                    f.unlink(missing_ok=True)

        except Exception as e:
            print(f"[cleanup] Error: {e}")

        await asyncio.sleep(interval_seconds)
