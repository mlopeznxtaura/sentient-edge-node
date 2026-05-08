"""
perception/vision.py
Sentient Edge Node — Cluster 01
YOLO-based object detection on local camera frames.
Outputs structured detection JSON for the agent loop.
"""

import cv2
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [vision] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    timestamp: float
    frame_id: int
    detections: list[Detection]
    inference_ms: float
    source: str


class VisionPipeline:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        source: int | str = 0,
        confidence_threshold: float = 0.4,
        device: str = "auto",
    ):
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.frame_id = 0

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        log.info(f"Loading YOLO model from {model_path} on {device}")

        self.model = YOLO(model_path)
        self.model.to(device)
        log.info("YOLO model loaded")

    def _parse_results(self, results, frame_shape) -> list[Detection]:
        h, w = frame_shape[:2]
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else None
                detections.append(Detection(
                    label=label,
                    confidence=round(conf, 3),
                    bbox=[round(x1/w, 4), round(y1/h, 4), round(x2/w, 4), round(y2/h, 4)],
                    track_id=track_id,
                ))
        return detections

    def run_once(self, frame) -> FrameResult:
        t0 = time.perf_counter()
        results = self.model.track(frame, persist=True, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000
        detections = self._parse_results(results, frame.shape)
        self.frame_id += 1
        return FrameResult(
            timestamp=time.time(),
            frame_id=self.frame_id,
            detections=detections,
            inference_ms=round(inference_ms, 2),
            source=str(self.source),
        )

    def stream(self, max_frames: int = 0):
        """Generator that yields FrameResult for each captured frame."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        log.info(f"Streaming from {self.source}")
        try:
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame read failed, retrying...")
                    time.sleep(0.05)
                    continue
                result = self.run_once(frame)
                log.info(
                    f"frame={result.frame_id} detections={len(result.detections)} "
                    f"inference={result.inference_ms}ms"
                )
                yield result
                count += 1
                if max_frames and count >= max_frames:
                    break
        finally:
            cap.release()
            log.info("Camera released")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    pipeline = VisionPipeline(
        model_path=args.model,
        source=args.source,
        confidence_threshold=args.conf,
    )
    for frame_result in pipeline.stream(max_frames=args.frames):
        print(json.dumps(asdict(frame_result), indent=2))
