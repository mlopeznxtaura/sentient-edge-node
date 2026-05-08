"""
agent/loop.py — Main perception-decision-action loop (Cluster 01: Sentient Edge Node)

Wires together: VisionPerception → AgentMemory → EdgePlanner → actuators.
Runs as the main process. All modules are hot-swappable via config reload.

SDKs: all cluster 01 SDKs wire through here
"""
import time
import json
import logging
import threading
import signal
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


class EdgeNodeLoop:
    """
    Main perception-decision-action loop.

    Tick rate is limited by YOLO inference speed (~30fps on GPU, ~5fps on CPU).
    Agent decisions run in a separate thread to avoid blocking vision.
    Actuator dispatch is async and non-blocking.
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._running = False

        # modules — loaded in start()
        self._vision = None
        self._depth = None
        self._audio = None
        self._memory = None
        self._planner = None
        self._mqtt = None
        self._zmq = None
        self._store = None
        self._metrics = None

        # agent decision throttle — don't call LLM every frame
        self._decision_interval_s = config.get("agent", {}).get("decision_interval_s", 1.0)
        self._last_decision_ts = 0.0
        self._decision_lock = threading.Lock()

    def _load_modules(self) -> None:
        from perception.vision import VisionPerception
        from perception.audio import AudioPerception
        from perception.depth import DepthPerception
        from store.db import EventStore
        from agent.memory import AgentMemory
        from agent.planner import EdgePlanner
        from actuator.mqtt_out import MQTTActuator
        from actuator.zmq_out import ZMQActuator
        from telemetry.metrics import EdgeMetrics

        logger.info("Loading modules...")
        self._store = EventStore(self._cfg)
        self._memory = AgentMemory(self._cfg, store=self._store)
        self._metrics = EdgeMetrics(self._cfg)

        self._vision = VisionPerception(self._cfg)
        self._vision.load()
        self._vision.open_camera()

        self._depth = DepthPerception(self._cfg)
        self._depth.load()

        self._audio = AudioPerception(self._cfg)
        self._audio.load()

        self._planner = EdgePlanner(self._cfg)
        self._planner.load()

        self._mqtt = MQTTActuator(self._cfg)
        self._mqtt.connect()

        self._zmq = ZMQActuator(self._cfg)
        self._zmq.connect()

        logger.info("All modules loaded")

    def _run_audio_thread(self) -> None:
        """Background thread: capture speech segments and push to memory."""
        try:
            for segment in self._audio.stream_segments():
                if not self._running:
                    break
                # In production: send audio bytes to Whisper or Deepgram here
                # For now, log segment arrival
                logger.debug("Speech segment: %.1fms, %d bytes", segment.duration_ms, len(segment.audio_bytes))
                self._memory.observe_speech(f"[speech segment {segment.segment_id}: {segment.duration_ms:.0f}ms]")
                if self._metrics:
                    self._metrics.speech_segments_total.inc()
        except Exception as e:
            logger.error("Audio thread error: %s", e)

    def _run_decision(self, frame_result, depth_map) -> None:
        """Run agent decision cycle (called in a separate thread)."""
        with self._decision_lock:
            context = self._memory.build_context()
            try:
                decision = self._planner.decide(context)
            except Exception as e:
                logger.error("Planner error: %s", e)
                return

            action = decision.get("action", "ignore")
            payload = decision.get("payload", {})
            reasoning = decision.get("reasoning", "")

            logger.info("Decision: %s | %s | %.1fms", action, reasoning, decision.get("decision_ms", 0))
            self._memory.record_decision(action, reasoning, payload)

            if self._store:
                self._store.log_decision(
                    timestamp_ms=time.time() * 1000,
                    inputs=context,
                    action=action,
                    reasoning=reasoning,
                    payload=payload,
                )

            self._dispatch_action(action, payload)
            if self._metrics:
                self._metrics.decisions_total.labels(action=action).inc()

    def _dispatch_action(self, action: str, payload: dict) -> None:
        """Route action to the correct actuator."""
        topic_prefix = self._cfg.get("mqtt", {}).get("topic_prefix", "edge/node01")

        if action == "alert":
            msg = {"action": "alert", **payload}
            self._mqtt.publish(f"{topic_prefix}/alerts", msg)
            self._zmq.send(msg)
        elif action == "track":
            self._mqtt.publish(f"{topic_prefix}/tracking", payload)
        elif action == "speak":
            self._mqtt.publish(f"{topic_prefix}/tts", payload)
        elif action == "log":
            note = payload.get("note", str(payload))
            self._memory.remember(f"note_{int(time.time())}", note)
        elif action == "ignore":
            pass

    def start(self) -> None:
        self._load_modules()
        self._running = True

        # start audio thread
        audio_thread = threading.Thread(target=self._run_audio_thread, daemon=True)
        audio_thread.start()

        # start metrics server
        if self._metrics:
            self._metrics.start_server()

        logger.info("Edge node loop started")
        frame_count = 0
        t_start = time.time()

        try:
            for frame_result in self._vision.stream():
                if not self._running:
                    break

                frame_count += 1
                now = time.time()

                # log each detection to store + memory
                for det in frame_result.detections:
                    depth_m = None
                    self._memory.observe_detection(det, depth_m=depth_m)
                    if self._store:
                        self._store.log_detection(
                            frame_id=frame_result.frame_id,
                            timestamp_ms=frame_result.timestamp_ms,
                            detection=det,
                            depth_m=depth_m,
                        )

                if self._metrics:
                    self._metrics.frames_total.inc()
                    self._metrics.detections_per_frame.observe(len(frame_result.detections))
                    self._metrics.inference_ms.observe(frame_result.inference_ms)

                # throttled agent decision
                if now - self._last_decision_ts >= self._decision_interval_s:
                    self._last_decision_ts = now
                    threading.Thread(
                        target=self._run_decision,
                        args=(frame_result, None),
                        daemon=True,
                    ).start()

                # FPS log every 100 frames
                if frame_count % 100 == 0:
                    elapsed = now - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info("FPS: %.1f | frame %d | dets: %d", fps, frame_count, len(frame_result.detections))

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._vision:
            self._vision.release()
        if self._audio:
            self._audio.release()
        if self._mqtt:
            self._mqtt.disconnect()
        if self._zmq:
            self._zmq.close()
        if self._store:
            self._store.close()
        logger.info("Edge node stopped")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    )
    parser = argparse.ArgumentParser(description="Sentient Edge Node")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    node = EdgeNodeLoop(cfg)

    def _handle_signal(sig, frame):
        logger.info("Signal %d received, shutting down...", sig)
        node.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    node.start()


if __name__ == "__main__":
    main()
