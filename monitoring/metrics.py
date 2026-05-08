"""
monitoring/metrics.py
Sentient Edge Node — Cluster 01
Prometheus metrics for the perception-decision-action loop.
"""

import time
import threading
from prometheus_client import Counter, Histogram, Gauge, start_http_server

frames_processed = Counter("edge_frames_total", "Total frames processed")
detections_total = Counter("edge_detections_total", "Total detections", ["label"])
inference_latency = Histogram("edge_inference_ms", "YOLO inference latency (ms)",
                               buckets=[10, 25, 50, 100, 200, 500, 1000])
agent_latency = Histogram("edge_agent_ms", "Agent reasoning latency (ms)",
                           buckets=[100, 500, 1000, 2000, 5000])
actions_dispatched = Counter("edge_actions_total", "Actions dispatched", ["action_type"])
mqtt_publishes = Counter("edge_mqtt_publishes_total", "MQTT messages published", ["topic"])
node_uptime = Gauge("edge_uptime_seconds", "Node uptime in seconds")

_start_time = time.time()


def _uptime_updater():
    while True:
        node_uptime.set(time.time() - _start_time)
        time.sleep(5)


def start_metrics_server(port: int = 8000):
    start_http_server(port)
    t = threading.Thread(target=_uptime_updater, daemon=True)
    t.start()
    import logging
    logging.getLogger("metrics").info(f"Prometheus metrics at http://0.0.0.0:{port}/metrics")
