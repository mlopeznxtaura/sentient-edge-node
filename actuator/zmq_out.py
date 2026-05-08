"""
actuator/zmq_out.py — ZeroMQ PUSH socket output (Cluster 01: Sentient Edge Node)

High-speed inter-process messaging via ZeroMQ PUSH socket.
Designed for low-latency actuator commands to co-located processes.

SDKs: pyzmq (ZeroMQ)
"""
import json
import logging
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class ZMQActuator:
    """
    ZeroMQ PUSH socket publisher.
    Non-blocking sends. Fire-and-forget for high-throughput actuator output.
    """

    def __init__(self, config: dict):
        cfg = config.get("zmq", {})
        self._address = cfg.get("push_address", "tcp://localhost:5555")
        self._hwm = cfg.get("hwm", 100)       # high water mark — drop old msgs if slow consumer
        self._socket = None
        self._ctx = None
        self._lock = threading.Lock()
        self._msg_count = 0

    def connect(self) -> None:
        try:
            import zmq
            self._ctx = zmq.Context()
            self._socket = self._ctx.socket(zmq.PUSH)
            self._socket.set_hwm(self._hwm)
            self._socket.connect(self._address)
            logger.info("ZMQ PUSH connected to %s", self._address)
        except ImportError:
            logger.warning("pyzmq not installed — ZMQ actuator disabled")
        except Exception as e:
            logger.warning("ZMQ connect failed: %s", e)

    def send(self, payload: dict) -> bool:
        if self._socket is None:
            return False
        with self._lock:
            try:
                import zmq
                self._socket.send_json(payload, flags=zmq.NOBLOCK)
                self._msg_count += 1
                logger.debug("ZMQ sent: %s", str(payload)[:80])
                return True
            except Exception as e:
                logger.debug("ZMQ send skipped: %s", e)
                return False

    def send_alert(self, message: str, severity: str = "medium") -> bool:
        return self.send({
            "type": "alert",
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
        })

    def send_command(self, command: str, params: Optional[dict] = None) -> bool:
        return self.send({
            "type": "command",
            "command": command,
            "params": params or {},
            "timestamp": time.time(),
        })

    @property
    def message_count(self) -> int:
        return self._msg_count

    def close(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._ctx:
            self._ctx.term()
            self._ctx = None
        logger.info("ZMQ socket closed")
