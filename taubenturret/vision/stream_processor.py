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

"""Processes video frames for the web livestream and overlays diagnostics."""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Handles frame formatting, overlay drawing, and MJPEG stream encoding."""

    def __init__(self) -> None:
        """Initialize the stream processor."""
        self.lock = Lock()
        self.latest_jpeg: bytes | None = None
        self.target_bbox: tuple[int, int, int, int] | None = None
        self.viewer_count: int = 0

        self._frame_to_encode: tuple[np.ndarray, dict[str, Any] | None, tuple[int, int, int, int] | None] | None = None
        self._new_frame_event = Event()
        self._encode_thread = Thread(target=self._encode_loop, daemon=True)
        self._encode_thread.start()

    def set_target(self, bbox: tuple[int, int, int, int] | None) -> None:
        """Thread-safely update the current AI target bounding box for the livestream overlay."""
        with self.lock:
            self.target_bbox = bbox

    def add_viewer(self) -> None:
        """Increment the active viewer count."""
        with self.lock:
            self.viewer_count += 1

    def remove_viewer(self) -> None:
        """Decrement the active viewer count."""
        with self.lock:
            self.viewer_count = max(0, self.viewer_count - 1)

    def process_frame(self, stream_bgr: np.ndarray | None, motion_region: dict[str, Any] | None) -> None:
        """
        Queue a raw camera frame for background overlay drawing and JPEG encoding.

        Intended to be called directly from the camera capture thread as a callback.
        """
        with self.lock:
            if self.viewer_count == 0 or stream_bgr is None:
                return

        t0 = time.perf_counter()

        with self.lock:
            self._frame_to_encode = (stream_bgr, motion_region, self.target_bbox)
        self._new_frame_event.set()

        logger.debug(f"Stream Profile (ms): async_dispatch={(time.perf_counter() - t0) * 1000:.1f}")

    def _encode_loop(self) -> None:
        """Background thread that handles the heavy JPEG compression without blocking the camera."""
        while True:
            self._new_frame_event.wait()
            self._new_frame_event.clear()

            with self.lock:
                if self._frame_to_encode is None:
                    continue
                stream_bgr, motion_region, target = self._frame_to_encode
                self._frame_to_encode = None

            if motion_region is not None:
                x1 = motion_region["x1"]
                y1 = motion_region["y1"]
                x2 = motion_region["x2"]
                y2 = motion_region["y2"]
                cv2.rectangle(
                    stream_bgr,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            if target is not None:
                x1, y1, x2, y2 = target
                cv2.rectangle(stream_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.drawMarker(
                    stream_bgr, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2
                )

            _, jpeg_buffer = cv2.imencode(".jpg", stream_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            with self.lock:
                self.latest_jpeg = jpeg_buffer.tobytes()

    def get_stream_jpeg(self) -> bytes | None:
        """Thread-safely get the latest JPEG buffer for the web livestream."""
        with self.lock:
            return self.latest_jpeg
