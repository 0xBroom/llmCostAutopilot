"""Async engine construction with the SQLite pragmas the write path needs.

Two pragmas, both about concurrent writers (the API serving a request and the
verification worker both write). WAL lets readers proceed while a writer holds
the log, and ``busy_timeout`` makes a second writer wait for the lock instead of
failing immediately with "database is locked". They are set per connection via a
``connect`` listener on this engine's sync core, not globally, so a second
engine (a test's throwaway DB) is unaffected.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# 5 seconds. Long enough that the worker waits out a normal request-path write
# rather than erroring; short enough that a genuinely stuck lock still surfaces.
_BUSY_TIMEOUT_MS = 5000


def create_engine(database_url: str) -> AsyncEngine:
    """Build the async engine and install the SQLite pragmas on every connection."""
    engine = create_async_engine(database_url)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        finally:
            cursor.close()

    return engine
