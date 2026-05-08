"""
api/routes.py — FastAPI route handlers (Cluster 01: Sentient Edge Node)

Exposes node status, recent events, memory, and manual override endpoints.
SDKs: FastAPI, Pydantic
"""
import time
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()
_node_ref = None  # injected by server.py


def set_node(node) -> None:
    global _node_ref
    _node_ref = node


# ── Request/Response models ───────────────────────────────────────────────────

class AlertRequest(BaseModel):
    message: str
    severity: str = "medium"

class MemorySetRequest(BaseModel):
    key: str
    value: object

class ActionRequest(BaseModel):
    action: str
    payload: dict = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@router.get("/status")
def status():
    if not _node_ref:
        return {"running": False}
    return {
        "running": _node_ref._running,
        "mqtt_connected": _node_ref._mqtt.is_connected if _node_ref._mqtt else False,
        "zmq_messages": _node_ref._zmq.message_count if _node_ref._zmq else 0,
        "mqtt_messages": _node_ref._mqtt.message_count if _node_ref._mqtt else 0,
    }

@router.get("/detections")
def recent_detections(limit: int = 20, class_name: Optional[str] = None):
    if not _node_ref or not _node_ref._store:
        raise HTTPException(503, "Store not available")
    return _node_ref._store.get_recent_detections(limit=limit, class_name=class_name)

@router.get("/memory")
def get_memory():
    if not _node_ref or not _node_ref._memory:
        raise HTTPException(503, "Memory not available")
    return _node_ref._memory.recall_all()

@router.post("/memory")
def set_memory(req: MemorySetRequest):
    if not _node_ref or not _node_ref._memory:
        raise HTTPException(503, "Memory not available")
    _node_ref._memory.remember(req.key, req.value)
    return {"ok": True, "key": req.key}

@router.post("/alert")
def trigger_alert(req: AlertRequest):
    if not _node_ref or not _node_ref._mqtt:
        raise HTTPException(503, "MQTT not available")
    sent = _node_ref._mqtt.publish_alert(req.message, req.severity)
    return {"sent": sent, "message": req.message, "severity": req.severity}

@router.post("/action")
def manual_action(req: ActionRequest, background_tasks: BackgroundTasks):
    if not _node_ref:
        raise HTTPException(503, "Node not available")
    background_tasks.add_task(_node_ref._dispatch_action, req.action, req.payload)
    return {"queued": True, "action": req.action}

@router.get("/context")
def get_context():
    if not _node_ref or not _node_ref._memory:
        raise HTTPException(503, "Memory not available")
    return _node_ref._memory.build_context()
