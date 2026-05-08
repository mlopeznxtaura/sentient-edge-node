"""
actuator/mqtt_publisher.py
Sentient Edge Node — Cluster 01
Publishes agent actions to MQTT broker (Mosquitto).
Runs on local network — no cloud dependency.
"""

import json
import logging
import time
import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)

TOPIC_ACTIONS = "edge/actions"
TOPIC_DETECTIONS = "edge/detections"
TOPIC_HEARTBEAT = "edge/heartbeat"


class MQTTPublisher:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        client_id: str = "sentient-edge-node",
        keepalive: int = 60,
    ):
        self.host = host
        self.port = port
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            log.info(f"Connected to MQTT broker at {self.host}:{self.port}")
        else:
            log.error(f"MQTT connect failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        log.warning(f"Disconnected from MQTT broker (rc={rc})")

    def connect(self, timeout: float = 5.0):
        self.client.connect(self.host, self.port)
        self.client.loop_start()
        deadline = time.time() + timeout
        while not self._connected and time.time() < deadline:
            time.sleep(0.1)
        if not self._connected:
            raise ConnectionError(f"Could not connect to MQTT broker at {self.host}:{self.port}")

    def publish_actions(self, actions: list[dict], frame_id: int):
        payload = json.dumps({"frame_id": frame_id, "actions": actions, "ts": time.time()})
        self.client.publish(TOPIC_ACTIONS, payload, qos=1)
        log.debug(f"Published {len(actions)} actions to {TOPIC_ACTIONS}")

    def publish_detections(self, frame_result: dict):
        payload = json.dumps(frame_result)
        self.client.publish(TOPIC_DETECTIONS, payload, qos=0)

    def heartbeat(self, status: str = "ok"):
        payload = json.dumps({"ts": time.time(), "status": status})
        self.client.publish(TOPIC_HEARTBEAT, payload, qos=0)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [mqtt] %(message)s")
    pub = MQTTPublisher()
    pub.connect()
    pub.publish_actions(
        actions=[{"type": "alert", "target": "person", "reason": "unknown individual detected"}],
        frame_id=42,
    )
    pub.heartbeat()
    pub.disconnect()
    print("Done.")
