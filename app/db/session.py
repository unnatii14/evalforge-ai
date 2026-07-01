import sqlite3
from pathlib import Path

from app.core.config import get_settings


def get_connection() -> sqlite3.Connection:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.sqlite_path)
    connection.row_factory = sqlite3.Row
    # Enforce foreign keys so ON DELETE CASCADE actually fires (off by default in SQLite).
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                chunk_size INTEGER NOT NULL,
                chunk_overlap INTEGER NOT NULL,
                top_k INTEGER NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                score REAL NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                contexts TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                answer_source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE
            )
            """
        )
        connection.commit()
