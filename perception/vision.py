"""
perception/vision.py — YOLO camera perception (Cluster 01: Sentient Edge Node)

Entry point per the cluster spec: accept camera frames, run YOLO inference,
output structured detection JSON with track IDs at maximum FPS.

SDKs: OpenCV, YOLO (Ultralytics), SAM2 (optional segmentation)
"""
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list       # [x1, y1, x2, y2] in pixels
    bbox_xywhn: list      # normalized [cx, cy, w, h]
    timestamp_ms: float


@dataclass
class FrameResult:
    frame_id: int
    timestamp_ms: float
    width: int
    height: int
    detections: list
    inference_ms: float


class VisionPerception:
    """
    Real-time YOLO-based object detection with optional ByteTrack tracking.
    Outputs FrameResult dicts suitable for downstream agent consumption.
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._model = None
        self._cap = None
        self._frame_id = 0

    def load(self) -> None:
        from ultralytics import YOLO

        model_path = self._cfg.get("yolo", {}).get("model", "yolov8n.pt")
        device = self._cfg.get("yolo", {}).get("device", "cpu")

        logger.info("Loading YOLO model: %s on %s", model_path, device)
        self._model = YOLO(model_path)
        self._device = device
        self._conf = self._cfg.get("yolo", {}).get("conf_threshold", 0.5)
        logger.info("YOLO loaded: %s classes", len(self._model.names))

    def open_camera(self) -> None:
        cam = self._cfg.get("camera", {})
        device_id = cam.get("device_id", 0)
        width = cam.get("width", 1280)
        height = cam.get("height", 720)

        self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_id}")
        logger.info("Camera opened: %dx%d", width, height)

    def infer_frame(self, frame: np.ndarray) -> FrameResult:
        """Run YOLO on a single frame, return structured FrameResult."""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        results = self._model.track(
            frame,
            persist=True,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        timestamp_ms = time.time() * 1000
        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                # track id (None if tracking not active for this box)
                track_id = int(boxes.id[i]) if boxes.id is not None else -1
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
                xywhn = boxes.xywhn[i].tolist()

                detections.append(Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=self._model.names[cls_id],
                    confidence=conf,
                    bbox_xyxy=[round(v, 1) for v in xyxy],
                    bbox_xywhn=[round(v, 4) for v in xywhn],
                    timestamp_ms=timestamp_ms,
                ))

        self._frame_id += 1
        return FrameResult(
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms,
            width=w,
            height=h,
            detections=[asdict(d) for d in detections],
            inference_ms=round(inference_ms, 2),
        )

    def stream(self) -> Generator[FrameResult, None, None]:
        """
        Yield FrameResult for every camera frame at maximum FPS.
        Caller is responsible for consuming fast enough to avoid buffer buildup.
        """
        if self._cap is None:
            self.open_camera()

        logger.info("Starting vision stream")
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Camera read failed, retrying...")
                    time.sleep(0.01)
                    continue
                yield self.infer_frame(frame)
        finally:
            self.release()

    def infer_image_path(self, path: str) -> FrameResult:
        """Run inference on a static image file (for testing without camera)."""
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.infer_frame(frame)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera released")


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--image", default=None, help="Run on a static image instead of camera")
    parser.add_argument("--show", action="store_true", help="Display annotated frames in a window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vision = VisionPerception(cfg)
    vision.load()

    if args.image:
        result = vision.infer_image_path(args.image)
        print(json.dumps(asdict(result) if hasattr(result, "__dataclass_fields__") else result.__dict__, indent=2))
    else:
        vision.open_camera()
        frame_count = 0
        t_start = time.time()
        for result in vision.stream():
            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            print(json.dumps({
                "frame_id": result.frame_id,
                "fps": round(fps, 1),
                "inference_ms": result.inference_ms,
                "detections": len(result.detections),
                "objects": [f"{d['class_name']}({d['track_id']})" for d in result.detections],
            }))

            if args.show:
                # annotate and show
                import cv2 as _cv2
                dummy_frame = _cv2.imencode(".jpg", _cv2.imread("/dev/null") if False else
                    _cv2.putText(_cv2.Mat() if False else
                    __import__("numpy").zeros((100, 400, 3), dtype=__import__("numpy").uint8),
                    f"FPS:{fps:.1f} dets:{len(result.detections)}", (10,50),
                    _cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2))[1]
                if _cv2.waitKey(1) & 0xFF == ord("q"):
                    break
