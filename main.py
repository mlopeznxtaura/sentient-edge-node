"""
main.py
Sentient Edge Node — Cluster 01
Entry point. Wires perception -> agent -> actuator into a running loop.
"""

import json
import logging
import signal
import sys
import time
from dataclasses import asdict

from perception.vision import VisionPipeline
from agent.brain import build_agent
from actuator.mqtt_publisher import MQTTPublisher
from monitoring.metrics import start_metrics_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("main")

RUNNING = True


def handle_shutdown(sig, frame):
    global RUNNING
    log.info("Shutdown signal received")
    RUNNING = False


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def run(
    camera_source=0,
    yolo_model="yolov8n.pt",
    llm_model="llama3",
    mqtt_host="localhost",
    confidence_threshold=0.4,
    metrics_port=8000,
):
    log.info("Starting Sentient Edge Node")

    # Boot components
    vision = VisionPipeline(
        model_path=yolo_model,
        source=camera_source,
        confidence_threshold=confidence_threshold,
    )
    agent = build_agent(model=llm_model)
    publisher = MQTTPublisher(host=mqtt_host)
    publisher.connect()
    start_metrics_server(port=metrics_port)

    log.info("All components initialized. Starting perception loop.")

    for frame_result in vision.stream():
        if not RUNNING:
            break

        # Convert to plain dict for agent
        detections = [asdict(d) for d in frame_result.detections]

        # Agent reasoning
        try:
            result = agent.invoke({
                "messages": [],
                "detections": detections,
                "actions": [],
                "cycle": 0,
            })
            actions = result.get("actions", [])
        except Exception as e:
            log.error(f"Agent error: {e}")
            actions = []

        # Publish
        publisher.publish_detections(asdict(frame_result))
        if actions:
            publisher.publish_actions(actions, frame_id=frame_result.frame_id)

        publisher.heartbeat()

    publisher.disconnect()
    log.info("Edge node stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentient Edge Node")
    parser.add_argument("--source", default=0)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--llm-model", default="llama3")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--metrics-port", type=int, default=8000)
    args = parser.parse_args()

    run(
        camera_source=args.source,
        yolo_model=args.yolo_model,
        llm_model=args.llm_model,
        mqtt_host=args.mqtt_host,
        confidence_threshold=args.conf,
        metrics_port=args.metrics_port,
    )
