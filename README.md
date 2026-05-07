# Cluster 01 — Sentient Edge Node

> Local AI agents that perceive, decide, and act without cloud dependency

## Architecture

```
Camera/Mic → perception/ → agent/loop.py → actuator/
                                ↓
                          store/db.py
                                ↓
                         telemetry/metrics.py
                                ↓
                          api/server.py
```

## 20 SDKs in this cluster

| SDK | Role |
|---|---|
| Ollama | Local LLM inference (llama3/mistral/phi) |
| llama.cpp | GGUF model loading fallback |
| LangGraph | Agent decision graph (stateful loops) |
| NVIDIA Warp | GPU-accelerated physics / simulation |
| OpenCV | Camera capture and image processing |
| YOLO (Ultralytics) | Real-time object detection |
| FastAPI | REST API for node control + event stream |
| SQLAlchemy | Local SQLite event store |
| Pydantic AI | Structured LLM output validation |
| DALI | GPU-accelerated data loading pipeline |
| MediaPipe | Hand/face/pose tracking |
| Mosquitto MQTT | Actuator output (IoT messaging) |
| TensorFlow Lite | Lightweight on-device inference |
| Open3D | Point cloud / 3D perception |
| Prometheus Client | Metrics export |
| PortAudio | Microphone capture |
| Depth Pro (Apple) | Monocular depth estimation |
| ZeroMQ | High-speed inter-process messaging |
| SAM2 | Zero-shot image segmentation |
| OpenTelemetry | Distributed tracing |

## Build order

1. `perception/vision.py` — YOLO on camera frames, structured JSON output
2. `perception/audio.py` — PortAudio mic capture, VAD, raw audio buffer
3. `perception/depth.py` — Depth Pro monocular depth from frame
4. `store/db.py` — SQLAlchemy event store
5. `agent/memory.py` — Short + long term memory backed by SQLite
6. `agent/planner.py` — LangGraph decision graph
7. `agent/loop.py` — Main perception-decision-action loop
8. `actuator/mqtt_out.py` — MQTT publisher for actuator commands
9. `actuator/zmq_out.py` — ZeroMQ PUSH socket for high-speed output
10. `api/routes.py` — FastAPI routes (status, events, override)
11. `api/server.py` — FastAPI app wiring
12. `telemetry/metrics.py` — Prometheus counters/gauges
13. `telemetry/tracing.py` — OpenTelemetry spans

## Quickstart

```bash
pip install -r requirements.txt
# Start Ollama first: ollama serve
python -m clusters.01_sentient_edge_node.agent.loop --config config.json
```
