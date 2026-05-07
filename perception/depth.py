"""
perception/depth.py — Monocular depth estimation (Cluster 01: Sentient Edge Node)

Uses Apple Depth Pro for metric depth from a single RGB frame.
Falls back to MiDaS (via torch.hub) if Depth Pro is unavailable.
Outputs a DepthResult with metric depth map and per-detection depth values.

SDKs: Depth Pro (Apple), OpenCV, PyTorch (fallback)
"""
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    frame_id: int
    timestamp_ms: float
    width: int
    height: int
    backend: str               # "depth_pro" or "midas"
    inference_ms: float
    focal_length_px: Optional[float]    # only from Depth Pro
    depth_min_m: float
    depth_max_m: float
    depth_mean_m: float
    depth_map: Optional[list] = None    # H x W float32 as nested list (omit for perf)


@dataclass
class DetectionDepth:
    track_id: int
    class_name: str
    depth_m: float             # median depth inside bbox
    depth_min_m: float
    depth_max_m: float


class DepthPerception:
    """
    Monocular depth estimator.
    Tries Depth Pro first (best accuracy, metric scale).
    Falls back to MiDaS (relative depth, normalized).
    """

    def __init__(self, config: dict):
        self._cfg = config.get("depth", {})
        self._backend = None
        self._model = None
        self._transform = None
        self._device = self._cfg.get("device", "cpu")
        self._include_map = self._cfg.get("include_map", False)

    def load(self) -> None:
        if self._try_load_depth_pro():
            self._backend = "depth_pro"
            logger.info("Depth Pro loaded")
        else:
            self._try_load_midas()
            self._backend = "midas"
            logger.info("MiDaS loaded (Depth Pro unavailable)")

    def _try_load_depth_pro(self) -> bool:
        try:
            import depth_pro
            self._model, self._transform = depth_pro.create_model_and_transforms(device=self._device)
            self._model.eval()
            return True
        except Exception as e:
            logger.debug("Depth Pro not available: %s", e)
            return False

    def _try_load_midas(self) -> None:
        import torch
        model_type = self._cfg.get("midas_model", "MiDaS_small")
        self._model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self._model.to(self._device)
        self._model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = (
            midas_transforms.dpt_transform
            if "DPT" in model_type
            else midas_transforms.small_transform
        )

    def infer_frame(self, frame: np.ndarray, frame_id: int = 0) -> DepthResult:
        """Run depth estimation on a single BGR frame."""
        h, w = frame.shape[:2]
        timestamp_ms = time.time() * 1000
        t0 = time.perf_counter()

        if self._backend == "depth_pro":
            result = self._infer_depth_pro(frame)
        else:
            result = self._infer_midas(frame)

        depth_map = result["depth_map"]
        inference_ms = (time.perf_counter() - t0) * 1000

        return DepthResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            width=w,
            height=h,
            backend=self._backend,
            inference_ms=round(inference_ms, 2),
            focal_length_px=result.get("focal_length_px"),
            depth_min_m=float(np.min(depth_map)),
            depth_max_m=float(np.max(depth_map)),
            depth_mean_m=float(np.mean(depth_map)),
            depth_map=depth_map.tolist() if self._include_map else None,
        ), depth_map

    def _infer_depth_pro(self, frame: np.ndarray) -> dict:
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._transform(rgb)
        with torch.no_grad():
            prediction = self._model.infer(image)
        depth = prediction["depth"].squeeze().cpu().numpy()
        focal = prediction.get("focallength_px")
        return {"depth_map": depth, "focal_length_px": float(focal) if focal is not None else None}

    def _infer_midas(self, frame: np.ndarray) -> dict:
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        # MiDaS is inverse depth — normalize to [0, 1] relative scale
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return {"depth_map": depth, "focal_length_px": None}

    def annotate_detections(self, detections: list, depth_map: np.ndarray) -> list:
        """
        For each detection bbox, compute median/min/max depth inside the box.
        Returns list of DetectionDepth dicts.
        """
        results = []
        h, w = depth_map.shape[:2] if depth_map.ndim == 2 else depth_map.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = depth_map[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            results.append(asdict(DetectionDepth(
                track_id=det["track_id"],
                class_name=det["class_name"],
                depth_m=round(float(np.median(roi)), 3),
                depth_min_m=round(float(np.min(roi)), 3),
                depth_max_m=round(float(np.max(roi)), 3),
            )))
        return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--include-map", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.include_map:
        cfg.setdefault("depth", {})["include_map"] = True

    depth_perc = DepthPerception(cfg)
    depth_perc.load()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Cannot read image: {args.image}")
        exit(1)

    result, depth_map = depth_perc.infer_frame(frame, frame_id=1)
    out = asdict(result)
    out.pop("depth_map", None)
    print(json.dumps(out, indent=2))
    print(f"Depth range: {result.depth_min_m:.3f}m — {result.depth_max_m:.3f}m")
