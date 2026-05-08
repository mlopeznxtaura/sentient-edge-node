# Sentient Edge Node

Local AI agents that perceive, decide, and act without cloud dependency.

**Cluster 01 of 25 — NextAura 500 SDKs / 25 Clusters**

## What This Is

An edge node that runs entirely on local hardware. No API keys. No cloud latency. No data leaving the machine.

The core is a perception-decision-action loop:
- Camera frames feed into YOLO for object detection and tracking
- A local LLM (via Ollama/LangGraph) reasons over detections and decides what to do
- Actions publish out over MQTT to physical systems or downstream consumers
- Prometheus exposes metrics for the whole loop

## Architecture

```
Camera → VisionPipeline (YOLO) → AgentBrain (LangGraph + Ollama) → MQTTPublisher → Actuators
                                                                  ↓
                                                           Prometheus /metrics
```

## 20 SDKs in This Cluster

Ollama · llama.cpp · LangGraph · NVIDIA Warp · OpenCV · YOLO (Ultralytics) · FastAPI · SQLAlchemy · Pydantic AI · DALI · MediaPipe · Mosquitto MQTT · TensorFlow Lite · Open3D · Prometheus Client · PortAudio · Depth Pro · ZeroMQ · SAM2 · OpenTelemetry

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull a local LLM via Ollama
ollama pull llama3

# 3. Start Mosquitto (MQTT broker)
mosquitto -d

# 4. Run the edge node
python main.py --source 0 --llm-model llama3

# Metrics available at:
# http://localhost:8000/metrics
```

## File Structure

```
sentient-edge-node/
├── perception/
│   └── vision.py          # YOLO detection + tracking pipeline
├── agent/
│   └── brain.py           # LangGraph decision agent (Ollama LLM)
├── actuator/
│   └── mqtt_publisher.py  # MQTT action dispatcher
├── monitoring/
│   └── metrics.py         # Prometheus metrics
├── main.py                # Entry point — wires it all together
└── requirements.txt
```

## Running in Docker

```bash
docker build -t sentient-edge-node .
docker run --gpus all --device /dev/video0 -p 8000:8000 sentient-edge-node
```

## Part of NextAura

github.com/mlopeznxtaura
