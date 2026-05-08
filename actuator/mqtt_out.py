"""
actuator/mqtt_out.py — MQTT publisher (Cluster 01: Sentient Edge Node)

Publishes agent action payloads to MQTT topics.
Uses Mosquitto/paho-mqtt. Thread-safe. Auto-reconnects.

SDKs: paho-mqtt (Mosquitto)
"""
import json
import logging
import time
import threading
from typing import Optional

import paho.mqtt.client as mqtt_client

logger = logging.getLogger(__name__)


class MQTTActuator:
    """
    MQTT publisher for edge node actuator output.
    Automatically reconnects on disconnect.
    All publishes are non-blocking (QoS 0 by default).
    """

    def __init__(self, config: dict):
        cfg = config.get("mqtt", {})
        self._broker = cfg.get("broker", "localhost")
        self._port = cfg.get("port", 1883)
        self._topic_prefix = cfg.get("topic_prefix", "edge/node01")
        self._qos = cfg.get("qos", 0)
        self._keepalive = cfg.get("keepalive", 60)
        self._client_id = cfg.get("client_id", f"edge-node-{int(time.time())}")
        self._username = cfg.get("username")
        self._password = cfg.get("password")
        self._client: Optional[mqtt_client.Client] = None
        self._connected = False
        self._lock = threading.Lock()
        self._msg_count = 0

    def connect(self) -> None:
        self._client = mqtt_client.Client(
            client_id=self._client_id,
            clean_session=True,
        )
        if self._username:
            self._client.username_pw_set(self._username, self._password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        try:
            self._client.connect(self._broker, self._port, self._keepalive)
            self._client.loop_start()
            logger.info("MQTT connecting to %s:%d", self._broker, self._port)
        except Exception as e:
            logger.warning("MQTT connect failed: %s (will retry on publish)", e)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to %s:%d", self._broker, self._port)
        else:
            logger.warning("MQTT connect refused, rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("MQTT unexpected disconnect rc=%d, reconnecting...", rc)

    def publish(self, topic: str, payload: dict, qos: Optional[int] = None) -> bool:
        """
        Publish a dict payload to a topic. Returns True if sent.
        Silently drops if broker unreachable (edge-safe behavior).
        """
        if self._client is None:
            logger.debug("MQTT client not initialized, skipping publish")
            return False

        msg = json.dumps(payload)
        effective_qos = qos if qos is not None else self._qos

        with self._lock:
            try:
                result = self._client.publish(topic, msg, qos=effective_qos)
                if result.rc == mqtt_client.MQTT_ERR_SUCCESS:
                    self._msg_count += 1
                    logger.debug("MQTT published to %s: %s", topic, msg[:80])
                    return True
                else:
                    logger.warning("MQTT publish failed rc=%d topic=%s", result.rc, topic)
                    return False
            except Exception as e:
                logger.error("MQTT publish error: %s", e)
                return False

    def publish_alert(self, message: str, severity: str = "medium", payload: Optional[dict] = None) -> bool:
        data = {"message": message, "severity": severity, "timestamp": time.time()}
        if payload:
            data.update(payload)
        return self.publish(f"{self._topic_prefix}/alerts", data, qos=1)

    def publish_detection(self, detection: dict) -> bool:
        return self.publish(f"{self._topic_prefix}/detections", detection)

    def publish_status(self, status: dict) -> bool:
        return self.publish(f"{self._topic_prefix}/status", status)

    @property
    def message_count(self) -> int:
        return self._msg_count

    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            self._connected = False
        logger.info("MQTT disconnected")
