"""Alembic environment configuration for Hydroponic Nutrition AI."""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import create_engine, pool

from alembic import context

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings
from backend.app.database.models import Base  # Registers all models

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    try:
        fileConfig(config.config_file_name, disable_existing_loggers=False)
    except (OSError, ValueError):
        sys.stderr.write("Notice: could not load alembic logging config\n")

target_metadata = Base.metadata


def get_database_url() -> str:
    """Return database URL from alembic -x argument, alembic config, or application settings."""
    x_args = context.get_x_argument(as_dictionary=True)
    if "url" in x_args:
        return x_args["url"]
    if "db_url" in x_args:
        return x_args["db_url"]

    url = config.get_main_option("sqlalchemy.url")
    if url and not url.startswith("driver://"):
        return url
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=url.startswith("sqlite"),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = get_database_url()
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=url.startswith("sqlite"),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
