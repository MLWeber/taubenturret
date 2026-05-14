# taubenturret - An automated, computer vision driven water turret system.
# Copyright (C) 2026 Michael Weber
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Coordinates the flow of video frames through the detection and recording sub-systems."""

import logging
import time
from threading import Lock
from typing import Any

import cv2
import numpy as np

from taubenturret import config
from taubenturret.vision.camera_controller import CameraController
from taubenturret.vision.motion_detector import MotionDetector
from taubenturret.vision.recorder import Recorder
from taubenturret.vision.stream_processor import StreamProcessor

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Glues the camera hardware, motion math, stream encoding, and MP4 recording together safely."""

    def __init__(self, camera_controller: CameraController) -> None:
        self.camera = camera_controller
        self.stream_processor = StreamProcessor()
        self.motion_detector = MotionDetector()
        self.recorder = Recorder()

        self.lock = Lock()
        self._latest_hires_bgr: np.ndarray | None = None
        self._latest_hires_region: dict[str, Any] | None = None
        self.is_active: bool = False

        self.t_last_motion: float = 0.0
        self.motion_off_delay = config.CAM_MOTION_OFF_DELAY

        self.camera.add_frame_callback(self.process_frame)

    def set_target(self, bbox: tuple[int, int, int, int] | None) -> None:
        """Pass the target bounding box to the stream processor for overlay."""
        if bbox is not None:
            scale_x = config.CAM_STREAM_RES[0] / config.CAM_RES[0]
            scale_y = config.CAM_STREAM_RES[1] / config.CAM_RES[1]
            stream_bbox = (
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y),
            )
            self.stream_processor.set_target(stream_bbox)
        else:
            self.stream_processor.set_target(None)

    def add_viewer(self) -> None:
        """Register an active viewer for the livestream."""
        self.stream_processor.add_viewer()

    def remove_viewer(self) -> None:
        """Unregister a viewer from the livestream."""
        self.stream_processor.remove_viewer()

    def process_frame(self, main_yuv: np.ndarray, lores_yuv: np.ndarray) -> None:
        """Callback executed for every frame captured by the camera."""
        sres_x, sres_y = config.CAM_STREAM_RES
        # Extract grayscale Y-channel directly (Numpy arrays are indexed as [height, width])
        lores_gray = lores_yuv[:sres_y, :sres_x]

        # Motion detector now internally resizes and returns coordinates scaled perfectly to CAM_STREAM_RES
        lores_bbox = self.motion_detector.detect(lores_gray)
        current_time = time.time()

        hires_bbox = None
        if lores_bbox is not None:
            self.t_last_motion = current_time

            rec_scale_x = config.CAM_RES[0] / config.CAM_STREAM_RES[0]
            rec_scale_y = config.CAM_RES[1] / config.CAM_STREAM_RES[1]
            hires_bbox = {
                "x1": int(lores_bbox["x1"] * rec_scale_x),
                "y1": int(lores_bbox["y1"] * rec_scale_y),
                "x2": int(lores_bbox["x2"] * rec_scale_x),
                "y2": int(lores_bbox["y2"] * rec_scale_y),
                "count": lores_bbox["count"],
            }

        is_active = current_time < self.t_last_motion + self.motion_off_delay
        has_viewers = self.stream_processor.viewer_count > 0

        hires_bgr = cv2.cvtColor(main_yuv, cv2.COLOR_YUV2BGR_I420) if is_active else None
        lores_bgr = cv2.cvtColor(lores_yuv, cv2.COLOR_YUV2BGR_I420) if has_viewers else None

        self.stream_processor.process_frame(lores_bgr, lores_bbox)

        with self.lock:
            self._latest_hires_bgr = hires_bgr
            self._latest_hires_region = hires_bbox if is_active else None
            self.is_active = is_active

        if is_active:
            if config.RECORD_ON_MOTION and not self.recorder.is_recording and hires_bgr is not None:
                self.recorder.start(self.camera.picam2, hires_bgr)
        else:
            self.recorder.stop(self.camera.picam2)

    def get_current_state(self) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        """Thread-safely get the latest high-res frame and its corresponding motion region atomically."""
        with self.lock:
            return self._latest_hires_bgr, self._latest_hires_region

    def get_stream_jpeg(self) -> bytes | None:
        """Get the latest JPEG frame for the livestream."""
        return self.stream_processor.get_stream_jpeg()

    def has_motion(self) -> bool:
        """Check if motion was detected within the active window."""
        with self.lock:
            return self.is_active

    def close(self) -> None:
        """Stop ongoing recordings and wait for them to save to disk."""
        self.recorder.stop(self.camera.picam2)
        self.recorder.close()
