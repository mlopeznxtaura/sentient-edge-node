"""
store/db.py — SQLAlchemy local event store (Cluster 01: Sentient Edge Node)

Persists perception events, agent decisions, and actuator commands
to a local SQLite database. Thread-safe. No cloud dependency.

SDKs: SQLAlchemy
"""
import json
import logging
import threading
from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, Boolean, Index, event as sa_event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(Integer, nullable=False)
    timestamp_ms = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=False)
    class_name = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_json = Column(Text, nullable=False)       # [x1,y1,x2,y2]
    depth_m = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("ix_detection_ts", "timestamp_ms"),
        Index("ix_detection_track", "track_id"),
    )


class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_ms = Column(Float, nullable=False)
    inputs_json = Column(Text, nullable=False)     # perception snapshot
    reasoning = Column(Text, nullable=True)        # LLM chain of thought
    action = Column(String(128), nullable=False)
    action_payload_json = Column(Text, nullable=True)
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("ix_decision_ts", "timestamp_ms"),
        Index("ix_decision_action", "action"),
    )


class ActuatorCommand(Base):
    __tablename__ = "actuator_commands"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_ms = Column(Float, nullable=False)
    channel = Column(String(64), nullable=False)   # "mqtt" or "zmq"
    topic = Column(String(256), nullable=True)
    payload_json = Column(Text, nullable=False)
    sent = Column(Boolean, default=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (Index("ix_actuator_ts", "timestamp_ms"),)


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(256), unique=True, nullable=False)
    value_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    __table_args__ = (Index("ix_memory_key", "key"),)


# ── Store ─────────────────────────────────────────────────────────────────────

class EventStore:
    """
    Thread-safe SQLAlchemy-backed local event store.
    One connection per thread via scoped_session.
    """

    def __init__(self, config: dict):
        db_path = config.get("db", {}).get("path", "edge_node.db")
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for concurrent reads + writes
        @sa_event.listens_for(self._engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")

        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        self._local = threading.local()
        logger.info("EventStore initialized: %s", db_path)

    def _session(self) -> Session:
        if not hasattr(self._local, "session") or self._local.session is None:
            self._local.session = self._Session()
        return self._local.session

    # ── Detection events ──────────────────────────────────────────────────────

    def log_detection(self, frame_id: int, timestamp_ms: float, detection: dict, depth_m: Optional[float] = None) -> int:
        sess = self._session()
        row = DetectionEvent(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            track_id=detection.get("track_id", -1),
            class_name=detection.get("class_name", "unknown"),
            confidence=detection.get("confidence", 0.0),
            bbox_json=json.dumps(detection.get("bbox_xyxy", [])),
            depth_m=depth_m,
        )
        sess.add(row)
        sess.commit()
        return row.id

    def get_recent_detections(self, limit: int = 100, class_name: Optional[str] = None) -> list:
        sess = self._session()
        q = sess.query(DetectionEvent).order_by(DetectionEvent.timestamp_ms.desc())
        if class_name:
            q = q.filter(DetectionEvent.class_name == class_name)
        rows = q.limit(limit).all()
        return [self._det_to_dict(r) for r in rows]

    def _det_to_dict(self, r: DetectionEvent) -> dict:
        return {
            "id": r.id, "frame_id": r.frame_id, "timestamp_ms": r.timestamp_ms,
            "track_id": r.track_id, "class_name": r.class_name,
            "confidence": r.confidence, "bbox_xyxy": json.loads(r.bbox_json),
            "depth_m": r.depth_m,
        }

    # ── Agent decisions ───────────────────────────────────────────────────────

    def log_decision(self, timestamp_ms: float, inputs: dict, action: str,
                     reasoning: Optional[str] = None, payload: Optional[dict] = None) -> int:
        sess = self._session()
        row = AgentDecision(
            timestamp_ms=timestamp_ms,
            inputs_json=json.dumps(inputs),
            reasoning=reasoning,
            action=action,
            action_payload_json=json.dumps(payload) if payload else None,
        )
        sess.add(row)
        sess.commit()
        return row.id

    def mark_decision_executed(self, decision_id: int) -> None:
        sess = self._session()
        row = sess.get(AgentDecision, decision_id)
        if row:
            row.executed = True
            sess.commit()

    # ── Actuator commands ─────────────────────────────────────────────────────

    def log_command(self, timestamp_ms: float, channel: str, payload: dict,
                    topic: Optional[str] = None) -> int:
        sess = self._session()
        row = ActuatorCommand(
            timestamp_ms=timestamp_ms,
            channel=channel,
            topic=topic,
            payload_json=json.dumps(payload),
        )
        sess.add(row)
        sess.commit()
        return row.id

    def mark_command_sent(self, cmd_id: int, error: Optional[str] = None) -> None:
        sess = self._session()
        row = sess.get(ActuatorCommand, cmd_id)
        if row:
            row.sent = error is None
            row.error = error
            sess.commit()

    # ── Memory ────────────────────────────────────────────────────────────────

    def memory_set(self, key: str, value) -> None:
        sess = self._session()
        row = sess.query(MemoryEntry).filter_by(key=key).first()
        if row:
            row.value_json = json.dumps(value)
            row.updated_at = datetime.now(UTC)
        else:
            sess.add(MemoryEntry(key=key, value_json=json.dumps(value)))
        sess.commit()

    def memory_get(self, key: str, default=None):
        sess = self._session()
        row = sess.query(MemoryEntry).filter_by(key=key).first()
        return json.loads(row.value_json) if row else default

    def memory_all(self) -> dict:
        sess = self._session()
        return {r.key: json.loads(r.value_json) for r in sess.query(MemoryEntry).all()}

    def close(self) -> None:
        if hasattr(self._local, "session") and self._local.session:
            self._local.session.close()
            self._local.session = None
        self._engine.dispose()
        logger.info("EventStore closed")


if __name__ == "__main__":
    import time, json, logging
    logging.basicConfig(level=logging.INFO)

    store = EventStore({"db": {"path": "/tmp/test_edge.db"}})

    # test detection log
    det_id = store.log_detection(
        frame_id=1,
        timestamp_ms=time.time() * 1000,
        detection={"track_id": 42, "class_name": "person", "confidence": 0.91, "bbox_xyxy": [100, 200, 300, 500]},
        depth_m=2.3,
    )
    print("logged detection:", det_id)

    # test memory
    store.memory_set("last_alert", {"class": "person", "ts": time.time()})
    print("memory get:", store.memory_get("last_alert"))

    # test decision
    dec_id = store.log_decision(
        timestamp_ms=time.time() * 1000,
        inputs={"detections": 1},
        action="alert",
        reasoning="Person detected in restricted zone",
        payload={"severity": "high"},
    )
    store.mark_decision_executed(dec_id)
    print("decision logged and marked executed:", dec_id)

    recent = store.get_recent_detections(limit=5)
    print("recent detections:", json.dumps(recent, indent=2))
    store.close()
