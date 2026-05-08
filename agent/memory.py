"""
agent/memory.py — Short and long-term memory (Cluster 01: Sentient Edge Node)

Short-term: in-process ring buffer of recent perceptions.
Long-term: SQLAlchemy-backed key/value store via store/db.py.

SDKs: SQLAlchemy (via store/db.py)
"""
import time
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    recent_detections: list      # last N detection events
    recent_decisions: list       # last N agent decisions
    persistent_keys: list        # all long-term memory keys


class ShortTermMemory:
    """Ring buffer of recent perception + decision events."""

    def __init__(self, maxlen: int = 64):
        self._detections: deque = deque(maxlen=maxlen)
        self._decisions: deque = deque(maxlen=maxlen)
        self._speech: deque = deque(maxlen=16)

    def add_detection(self, det: dict) -> None:
        self._detections.append({**det, "_added_at": time.time()})

    def add_decision(self, dec: dict) -> None:
        self._decisions.append({**dec, "_added_at": time.time()})

    def add_speech(self, text: str) -> None:
        self._speech.append({"text": text, "_added_at": time.time()})

    def recent_detections(self, n: int = 10) -> list:
        return list(self._detections)[-n:]

    def recent_decisions(self, n: int = 5) -> list:
        return list(self._decisions)[-n:]

    def recent_speech(self, n: int = 5) -> list:
        return list(self._speech)[-n:]

    def classes_seen(self) -> dict:
        """Return count of each class seen in recent detections."""
        counts = {}
        for d in self._detections:
            name = d.get("class_name", "unknown")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def track_ids_active(self) -> set:
        """Return set of track_ids seen in the last 2 seconds."""
        now = time.time()
        return {d["track_id"] for d in self._detections
                if now - d.get("_added_at", 0) < 2.0 and d.get("track_id", -1) >= 0}

    def clear(self) -> None:
        self._detections.clear()
        self._decisions.clear()
        self._speech.clear()


class AgentMemory:
    """
    Combined short + long-term memory for the agent loop.
    Short-term is in-process. Long-term persists via EventStore.
    """

    def __init__(self, config: dict, store=None):
        self._cfg = config.get("memory", {})
        maxlen = self._cfg.get("short_term_maxlen", 64)
        self.short = ShortTermMemory(maxlen=maxlen)
        self._store = store  # EventStore instance, injected

    def observe_detection(self, det: dict, depth_m: Optional[float] = None) -> None:
        """Record a detection into short-term memory."""
        record = {**det}
        if depth_m is not None:
            record["depth_m"] = depth_m
        self.short.add_detection(record)

    def observe_speech(self, text: str) -> None:
        self.short.add_speech(text)

    def record_decision(self, action: str, reasoning: str, payload: dict) -> None:
        self.short.add_decision({
            "action": action,
            "reasoning": reasoning,
            "payload": payload,
            "timestamp_ms": time.time() * 1000,
        })

    # ── Long-term (persistent) ────────────────────────────────────────────────

    def remember(self, key: str, value: Any) -> None:
        """Persist a fact to long-term memory."""
        if self._store:
            self._store.memory_set(key, value)
        logger.debug("remember: %s = %s", key, value)

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact from long-term memory."""
        if self._store:
            return self._store.memory_get(key, default)
        return default

    def recall_all(self) -> dict:
        if self._store:
            return self._store.memory_all()
        return {}

    # ── Context builder ───────────────────────────────────────────────────────

    def build_context(self) -> dict:
        """
        Build a structured context dict for the LLM agent.
        Keeps token count bounded by summarizing rather than dumping raw history.
        """
        classes = self.short.classes_seen()
        active_tracks = list(self.short.track_ids_active())
        recent_dets = self.short.recent_detections(n=5)
        recent_speech = self.short.recent_speech(n=3)
        recent_decisions = self.short.recent_decisions(n=3)

        # Prune heavy fields for context
        def _slim(d: dict) -> dict:
            return {k: v for k, v in d.items()
                    if k not in ("bbox_xyxy", "bbox_xywhn", "_added_at")}

        return {
            "timestamp": time.time(),
            "active_track_ids": active_tracks,
            "classes_seen": classes,
            "recent_detections": [_slim(d) for d in recent_dets],
            "recent_speech": [s["text"] for s in recent_speech],
            "recent_decisions": [_slim(d) for d in recent_decisions],
            "long_term": self.recall_all(),
        }

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            recent_detections=self.short.recent_detections(),
            recent_decisions=self.short.recent_decisions(),
            persistent_keys=list(self.recall_all().keys()),
        )
