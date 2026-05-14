"""Background subtraction and motion detection logic."""

import logging
import time
from typing import Any

import cv2
import numpy as np

from taubenturret import config

logger = logging.getLogger(__name__)


class MotionDetector:
    """Applies background subtraction to identify moving targets."""

    def __init__(self) -> None:
        stream_w, stream_h = config.CAM_STREAM_RES
        motion_w = stream_w * config.CAM_MOTION_RES_FACTOR
        motion_h = stream_h * config.CAM_MOTION_RES_FACTOR

        self.min_area = (config.CAM_MOTION_MIN_AREA / 100.0) * motion_w * motion_h
        self.threshold = config.CAM_MOTION_THRESHOLD
        self.blur_size = (5, 5)
        self.bg_learning_rate = config.CAM_MOTION_BG_LEARNING_RATE

        self.bg_model: np.ndarray | None = None
        self.warmup_delay = config.CAM_MOTION_OFF_DELAY

        self.start_time = time.time()

    def detect(self, frame_gray: np.ndarray) -> dict[str, Any] | None:
        """Apply background subtraction and return the scaled bounding box if motion is present."""
        current_time = time.time()

        # Scale down internally for faster background subtraction processing
        h, w = frame_gray.shape[:2]
        small_w = int(w * config.CAM_MOTION_RES_FACTOR)
        small_h = int(h * config.CAM_MOTION_RES_FACTOR)
        small_gray = cv2.resize(frame_gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        # Box blur is heavily SIMD-optimized and much faster than Gaussian on ARM CPUs
        gray = cv2.blur(small_gray, self.blur_size)

        if self.bg_model is None:
            # float32 runs ~4x faster than default float64 on Pi Zero 2's NEON architecture
            self.bg_model = gray.astype(np.float32)
            return None

        # Update the background model and calculate difference
        cv2.accumulateWeighted(gray, self.bg_model, self.bg_learning_rate)

        # Ignore motion during the warmup phase so AE/AWB can stabilize
        if current_time - self.start_time < self.warmup_delay:
            return None

        frame_diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg_model))
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)  # type: ignore[call-overload]

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area > self.min_area:
                valid_contours.append((c, area))

        if valid_contours:
            best_contour, max_area = max(valid_contours, key=lambda item: item[1])
            bx, by, bw, bh = cv2.boundingRect(best_contour)

            scale_up = 1.0 / config.CAM_MOTION_RES_FACTOR
            return {
                "x1": int(bx * scale_up),
                "y1": int(by * scale_up),
                "x2": int((bx + bw) * scale_up),
                "y2": int((by + bh) * scale_up),
                "count": max_area * (scale_up**2),
            }

        return None
