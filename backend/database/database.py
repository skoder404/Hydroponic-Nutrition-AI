import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                ph REAL,
                ec REAL,
                water_temp REAL,
                humidity REAL,
                air_temp REAL,
                image_path TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id INTEGER NOT NULL,
                sensor_prediction TEXT,
                sensor_confidence REAL,
                sensor_probabilities TEXT,
                image_prediction TEXT,
                image_confidence REAL,
                image_probabilities TEXT,
                fusion_status TEXT,
                final_diagnosis TEXT,
                final_confidence REAL,
                recommendation TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(observation_id) REFERENCES observations(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_observation(
    *,
    source: str,
    ph: Optional[float] = None,
    ec: Optional[float] = None,
    water_temp: Optional[float] = None,
    humidity: Optional[float] = None,
    air_temp: Optional[float] = None,
    image_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> int:
    conn = get_connection()
    try:
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cursor = conn.execute(
            """
            INSERT INTO observations (
                timestamp, source, ph, ec, water_temp, humidity, air_temp, image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, source, ph, ec, water_temp, humidity, air_temp, image_path),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def save_prediction(
    *,
    observation_id: int,
    sensor_prediction: Optional[str] = None,
    sensor_confidence: Optional[float] = None,
    sensor_probabilities: Optional[Dict[str, float]] = None,
    image_prediction: Optional[str] = None,
    image_confidence: Optional[float] = None,
    image_probabilities: Optional[Dict[str, float]] = None,
    fusion_status: Optional[str] = None,
    final_diagnosis: Optional[str] = None,
    final_confidence: Optional[float] = None,
    recommendation: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> int:
    conn = get_connection()
    try:
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                observation_id,
                sensor_prediction,
                sensor_confidence,
                sensor_probabilities,
                image_prediction,
                image_confidence,
                image_probabilities,
                fusion_status,
                final_diagnosis,
                final_confidence,
                recommendation,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                sensor_prediction,
                sensor_confidence,
                json.dumps(sensor_probabilities) if sensor_probabilities is not None else None,
                image_prediction,
                image_confidence,
                json.dumps(image_probabilities) if image_probabilities is not None else None,
                fusion_status,
                final_diagnosis,
                final_confidence,
                recommendation,
                timestamp,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def save_analysis(
    *,
    source: str,
    sensor_payload: Dict[str, float],
    sensor_result: Dict[str, Any],
    image_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
    recommendation: Dict[str, Any],
    image_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> int:
    conn = get_connection()
    try:
        observed_at = timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
        cursor = conn.execute(
            """
            INSERT INTO observations (
                timestamp, source, ph, ec, water_temp, humidity, air_temp, image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observed_at,
                source,
                sensor_payload["ph"],
                sensor_payload["ec"],
                sensor_payload["water_temp"],
                sensor_payload["humidity"],
                sensor_payload["air_temp"],
                image_path,
            ),
        )
        observation_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO predictions (
                observation_id, sensor_prediction, sensor_confidence,
                sensor_probabilities, image_prediction, image_confidence,
                image_probabilities, fusion_status, final_diagnosis,
                final_confidence, recommendation, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                sensor_result.get("prediction"),
                sensor_result.get("confidence"),
                json.dumps(sensor_result.get("probabilities", {})),
                image_result.get("predicted_class"),
                image_result.get("confidence"),
                json.dumps(image_result.get("class_probabilities", {})),
                fusion_result.get("status"),
                fusion_result.get("final_diagnosis"),
                fusion_result.get("confidence"),
                json.dumps(recommendation),
                observed_at,
            ),
        )
        conn.commit()
        return observation_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_latest_sensor_reading() -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT o.*, p.sensor_prediction, p.sensor_confidence, p.fusion_status
            FROM observations o
            LEFT JOIN predictions p ON p.observation_id = o.id
            ORDER BY o.id DESC LIMIT 1"""
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT
                o.id AS observation_id, o.timestamp AS observed_at, o.source,
                o.ph, o.ec, o.water_temp, o.humidity, o.air_temp, o.image_path,
                p.id AS prediction_id, p.sensor_prediction, p.sensor_confidence,
                p.sensor_probabilities, p.image_prediction, p.image_confidence,
                p.image_probabilities, p.fusion_status, p.final_diagnosis,
                p.final_confidence, p.recommendation, p.timestamp AS predicted_at
            FROM observations o
            LEFT JOIN predictions p ON p.observation_id = o.id
            ORDER BY o.id DESC
            LIMIT ?""",
            (limit,),
        ).fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            for key in ("sensor_probabilities", "image_probabilities", "recommendation"):
                value = item.get(key)
                if value:
                    try:
                        item[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
        return items
    finally:
        conn.close()
