"""
sim_runner.py — Simulation harness for Sentient Edge Node

Runs the full perception-decision-action loop with:
- Synthetic camera frames (random detections, no real camera)
- Synthetic audio (random speech segments)
- Mocked MQTT/ZMQ (no broker needed)
- Real LLM decisions via Ollama (falls back to rule-based if unavailable)

Logs every iteration to iterations.jsonl in the format:
  {iter, timestamp_ms, frame, detections, decision, action, payload, reasoning, actuator_outputs, memory_snapshot}
"""
import time
import json
import random
import logging
import os
import sys
import threading
import math
from dataclasses import dataclass, asdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s"
)
logger = logging.getLogger("sim")

OUTPUT_PATH = os.environ.get("SIM_OUTPUT", "/opt/data/sim/iterations.jsonl")
MAX_ITERS = int(os.environ.get("SIM_ITERS", "50"))
DECISION_INTERVAL = float(os.environ.get("SIM_DECISION_INTERVAL", "1.0"))
USE_OLLAMA = os.environ.get("SIM_USE_OLLAMA", "0") == "1"

CLASSES = ["person", "car", "bicycle", "dog", "laptop", "chair", "bottle", "phone"]
ZONES = ["entry", "restricted", "public", "perimeter"]
ACTIONS = ["alert", "track", "ignore", "log", "speak"]
SEVERITIES = ["low", "medium", "high", "critical"]

# ── Synthetic generators ───────────────────────────────────────────────────────

def synthetic_frame(frame_id: int) -> dict:
    """Generate a realistic-looking camera frame with detections."""
    n_dets = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 25, 15, 5])[0]
    detections = []
    for i in range(n_dets):
        cls = random.choice(CLASSES)
        x1 = random.randint(0, 1100)
        y1 = random.randint(0, 620)
        x2 = x1 + random.randint(40, 200)
        y2 = y1 + random.randint(40, 200)
        detections.append({
            "track_id": random.randint(1, 20),
            "class_id": CLASSES.index(cls),
            "class_name": cls,
            "confidence": round(random.uniform(0.5, 0.99), 3),
            "bbox_xyxy": [x1, y1, min(x2, 1280), min(y2, 720)],
            "bbox_xywhn": [
                round((x1 + x2) / 2 / 1280, 4),
                round((y1 + y2) / 2 / 720, 4),
                round((x2 - x1) / 1280, 4),
                round((y2 - y1) / 720, 4),
            ],
            "depth_m": round(random.uniform(0.5, 8.0), 2),
            "timestamp_ms": time.time() * 1000,
        })
    return {
        "frame_id": frame_id,
        "timestamp_ms": time.time() * 1000,
        "width": 1280,
        "height": 720,
        "detections": detections,
        "inference_ms": round(random.uniform(8, 35), 2),
    }


def synthetic_audio_event() -> Optional[dict]:
    """Occasionally generate a speech segment."""
    if random.random() > 0.15:
        return None
    phrases = [
        "motion detected near the entry",
        "all clear zone two",
        "unknown object approaching",
        "system check nominal",
        "override acknowledged",
    ]
    return {
        "segment_id": random.randint(1, 999),
        "text": random.choice(phrases),
        "duration_ms": round(random.uniform(800, 3000), 1),
    }


def rule_based_decision(context: dict) -> dict:
    """Fast rule-based fallback when Ollama is not available."""
    detections = context.get("recent_detections", [])
    classes = context.get("classes_seen", {})
    zone = context.get("long_term", {}).get("zone", "public")

    if "person" in classes and zone == "restricted":
        return {
            "action": "alert",
            "payload": {"message": "Person in restricted zone", "severity": "high"},
            "reasoning": "Person detected in restricted zone — escalating alert",
        }
    elif len(detections) > 3:
        return {
            "action": "track",
            "payload": {"track_id": detections[0].get("track_id", 1), "label": detections[0].get("class_name", "object")},
            "reasoning": "Multiple objects detected — focusing tracking on primary",
        }
    elif not detections:
        return {
            "action": "ignore",
            "payload": {},
            "reasoning": "No detections this cycle",
        }
    else:
        cls = detections[0].get("class_name", "object")
        return {
            "action": "log",
            "payload": {"note": f"{cls} observed at depth {detections[0].get('depth_m', '?')}m"},
            "reasoning": f"Routine {cls} observation — logging for record",
        }


def ollama_decision(context: dict, model: str = "llama3") -> dict:
    """Call Ollama for a real LLM decision."""
    import urllib.request
    prompt = f"""You are an edge AI node. Given this perception context, decide one action.
Context: {json.dumps(context, indent=2)}

Reply ONLY with JSON: {{"action": "alert|track|ignore|log|speak", "payload": {{}}, "reasoning": "one sentence"}}"""
    payload = {"model": model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150}}
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            resp = json.loads(r.read())
            return json.loads(resp["response"])
    except Exception as e:
        logger.debug("Ollama unavailable (%s), using rule-based", e)
        return rule_based_decision(context)


# ── Sim memory ─────────────────────────────────────────────────────────────────

class SimMemory:
    def __init__(self):
        self._short = []
        self._long = {
            "zone": random.choice(ZONES),
            "node_id": "sim-node-01",
            "alert_cooldown_s": 10,
            "session_start": time.time(),
        }
        self._decisions = []
        self._speech = []

    def add_detection(self, det): self._short.append(det)
    def add_speech(self, text): self._speech.append({"text": text, "ts": time.time()})
    def add_decision(self, d): self._decisions.append(d)
    def remember(self, k, v): self._long[k] = v
    def recall(self, k, default=None): return self._long.get(k, default)

    def build_context(self) -> dict:
        recent = self._short[-10:]
        classes = {}
        for d in recent:
            cls = d.get("class_name", "unknown")
            classes[cls] = classes.get(cls, 0) + 1
        return {
            "timestamp": time.time(),
            "active_track_ids": list({d.get("track_id", -1) for d in recent}),
            "classes_seen": classes,
            "recent_detections": [{k: v for k, v in d.items() if k not in ("timestamp_ms", "bbox_xywhn")}
                                   for d in recent[-5:]],
            "recent_speech": [s["text"] for s in self._speech[-3:]],
            "recent_decisions": [{"action": d["action"], "reasoning": d["reasoning"]}
                                  for d in self._decisions[-3:]],
            "long_term": self._long,
        }

    def snapshot(self) -> dict:
        return {
            "short_term_count": len(self._short),
            "decision_count": len(self._decisions),
            "speech_count": len(self._speech),
            "long_term_keys": list(self._long.keys()),
            "zone": self._long.get("zone"),
        }


# ── Sim actuator log ───────────────────────────────────────────────────────────

class SimActuator:
    def __init__(self):
        self.log = []

    def dispatch(self, action: str, payload: dict) -> dict:
        record = {
            "channel": "mqtt" if action in ("alert", "track", "speak") else "internal",
            "action": action,
            "payload": payload,
            "timestamp_ms": time.time() * 1000,
            "simulated": True,
        }
        self.log.append(record)
        return record

    def flush(self) -> list:
        out = list(self.log)
        self.log.clear()
        return out


# ── Main simulation loop ───────────────────────────────────────────────────────

def run_simulation():
    logger.info("Starting simulation | iters=%d | output=%s", MAX_ITERS, OUTPUT_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    memory = SimMemory()
    actuator = SimActuator()

    last_decision_ts = 0.0
    iter_count = 0
    frame_id = 0

    with open(OUTPUT_PATH, "w") as out_f:
        while iter_count < MAX_ITERS:
            frame_id += 1
            iter_count += 1
            now = time.time()

            # 1. Perception
            frame = synthetic_frame(frame_id)
            for det in frame["detections"]:
                memory.add_detection(det)

            # 2. Optional speech
            speech = synthetic_audio_event()
            if speech:
                memory.add_speech(speech["text"])

            # 3. Agent decision (throttled)
            decision = None
            if now - last_decision_ts >= DECISION_INTERVAL:
                last_decision_ts = now
                context = memory.build_context()
                if USE_OLLAMA:
                    raw = ollama_decision(context)
                else:
                    raw = rule_based_decision(context)

                decision = {
                    "action": raw.get("action", "ignore"),
                    "payload": raw.get("payload", {}),
                    "reasoning": raw.get("reasoning", ""),
                    "decision_ms": round(random.uniform(5, 25), 1),
                    "context_snapshot": context,
                }
                memory.add_decision(decision)

                # 4. Actuator dispatch
                actuator.dispatch(decision["action"], decision["payload"])

            # 5. Write JSONL record
            record = {
                "iter": iter_count,
                "frame_id": frame_id,
                "timestamp_ms": round(now * 1000, 1),
                "frame": {
                    "inference_ms": frame["inference_ms"],
                    "detection_count": len(frame["detections"]),
                    "detections": frame["detections"],
                },
                "speech": speech,
                "decision": decision,
                "actuator_outputs": actuator.flush(),
                "memory_snapshot": memory.snapshot(),
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if iter_count % 10 == 0:
                logger.info("iter %d/%d | frame %d | dets: %d | decision: %s",
                            iter_count, MAX_ITERS, frame_id,
                            len(frame["detections"]),
                            decision["action"] if decision else "—")

            # ~30fps simulation speed (no real sleep needed for logs)
            time.sleep(0.033)

    logger.info("Simulation complete. %d iterations written to %s", iter_count, OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = run_simulation()
    # print last 3 records as sample
    lines = open(path).readlines()
    print(f"\n=== Sample (last 3 of {len(lines)} records) ===")
    for line in lines[-3:]:
        rec = json.loads(line)
        print(json.dumps({
            "iter": rec["iter"],
            "dets": rec["frame"]["detection_count"],
            "decision": rec["decision"]["action"] if rec["decision"] else None,
            "reasoning": rec["decision"]["reasoning"][:60] if rec["decision"] else None,
            "actuator_outputs": len(rec["actuator_outputs"]),
        }, indent=2))
