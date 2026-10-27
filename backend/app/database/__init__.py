"""Database package initialization exporting Base, session utilities, and ORM models."""
from __future__ import annotations

from backend.app.database.models import (
    AnalysisRecord,
    Input,
    Prediction,
    Recommendation,
)
from backend.app.database.repositories import (
    PersistenceError,
    RepositoryError,
    SaveResult,
    create_record,
    get_original_observed_datetime,
    get_record,
    list_records,
    reconstruct_observed_datetime,
)
from backend.app.database.session import (
    Base,
    get_db,
    get_engine,
    get_session_factory,
)

__all__ = [
    "AnalysisRecord",
    "Base",
    "Input",
    "PersistenceError",
    "Prediction",
    "Recommendation",
    "RepositoryError",
    "SaveResult",
    "create_record",
    "get_db",
    "get_engine",
    "get_original_observed_datetime",
    "get_record",
    "get_session_factory",
    "list_records",
    "reconstruct_observed_datetime",
]
