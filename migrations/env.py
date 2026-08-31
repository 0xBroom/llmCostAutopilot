"""Alembic environment.

The database URL is read from the application's own ``Settings`` rather than
hard-coded in ``alembic.ini``, so migrations always target the same SQLite file
the process serves from. ``target_metadata`` points at the Core metadata in
``autopilot.infrastructure.persistence.schema`` so ``--autogenerate`` sees the
tables owned by this codebase.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from autopilot.config.settings import load_settings
from autopilot.infrastructure.persistence.engine import create_sqlite_engine
from autopilot.infrastructure.persistence.schema import metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Single source of truth for where the database lives.
config.set_main_option("sqlalchemy.url", load_settings().database_url)

target_metadata = metadata


def run_migrations_offline() -> None:
    """Emit SQL against a URL, without a live DBAPI connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through a sync-bridged connection."""
    connectable = create_sqlite_engine(load_settings().database_url)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
