"""
telemetry/metrics.py — Prometheus metrics (Cluster 01: Sentient Edge Node)

Exposes node telemetry: frames processed, detections, inference latency,
decision counts, MQTT/ZMQ message counts.

SDKs: prometheus-client
"""
import logging
import threading
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, REGISTRY
)

logger = logging.getLogger(__name__)


class EdgeMetrics:
    def __init__(self, config: dict):
        self._port = config.get("telemetry", {}).get("prometheus_port", 9090)
        self._server_started = False

        self.frames_total = Counter(
            "edge_frames_total", "Total camera frames processed"
        )
        self.detections_per_frame = Histogram(
            "edge_detections_per_frame", "Number of detections per frame",
            buckets=[0, 1, 2, 5, 10, 20, 50]
        )
        self.inference_ms = Histogram(
            "edge_inference_ms", "YOLO inference latency in milliseconds",
            buckets=[5, 10, 20, 50, 100, 200, 500]
        )
        self.decisions_total = Counter(
            "edge_decisions_total", "Agent decisions by action",
            labelnames=["action"]
        )
        self.speech_segments_total = Counter(
            "edge_speech_segments_total", "Total speech segments detected"
        )
        self.mqtt_published_total = Counter(
            "edge_mqtt_published_total", "MQTT messages published"
        )
        self.zmq_sent_total = Counter(
            "edge_zmq_sent_total", "ZMQ messages sent"
        )
        self.active_tracks = Gauge(
            "edge_active_tracks", "Number of currently tracked objects"
        )
        self.loop_fps = Gauge(
            "edge_loop_fps", "Current perception loop FPS"
        )

    def start_server(self) -> None:
        if not self._server_started:
            try:
                start_http_server(self._port)
                self._server_started = True
                logger.info("Prometheus metrics at http://0.0.0.0:%d/metrics", self._port)
            except Exception as e:
                logger.warning("Could not start Prometheus server: %s", e)
