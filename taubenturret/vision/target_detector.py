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

"""Handles communication with the external AI object detection backend."""

import logging
import time
import urllib.parse
from dataclasses import dataclass

import cv2
import numpy as np
import requests

from taubenturret import config

logger: logging.Logger = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)


@dataclass
class Target:
    """Represents a detected object with its pixel boundaries and physical targeting angles."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return (round((self.x1 + self.x2) / 2), round((self.y1 + self.y2) / 2))


class TargetDetector:
    """Service class for dispatching images to a remote API for object detection."""

    def __init__(self) -> None:
        """
        Initialize the TargetDetector, establish an HTTP session, and ping the
        detection API to verify connectivity.
        """
        self.http_session: requests.Session = requests.Session()
        self._validate_detector_classes()

    def _validate_detector_classes(self) -> None:
        try:
            res: requests.Response = self.http_session.get(config.DETECTOR_API_URL + "ping")
            if res.status_code == 200:
                logger.info("Successfully pinged detection API.")
            else:
                logger.warning(f"Failed to ping detection API: {res.status_code}")

            classes_res: requests.Response = self.http_session.get(config.DETECTOR_API_URL + "classes")
            if classes_res.status_code == 200:
                supported_classes = classes_res.json()["classes"]

                target_classes = urllib.parse.unquote(config.DETECTOR_TARGET_CLASSES).split(",")
                for tc in target_classes:
                    tc = urllib.parse.quote(tc.strip().lower())
                    if tc and tc not in supported_classes:
                        logger.warning(f"Configured class '{tc}' is not supported by the backend!")
            else:
                logger.warning(f"Could not fetch supported classes for validation: {classes_res.status_code}")
        except Exception:
            logger.exception("Could not fetch supported classes for validation.")

    def _prepare_image(self, frame: np.ndarray, motion_region: dict[str, int] | None) -> tuple[np.ndarray, int, int]:
        """
        Determine if the frame should be cropped based on the motion region and backend constraints.

        Returns:
            A tuple containing (cropped_image_array, x_offset, y_offset).
        """
        img = frame
        x_offset: int = 0
        y_offset: int = 0
        if (
            motion_region
            and (motion_region["x2"] - motion_region["x1"]) * config.DETECTOR_CROP_PADDING_FACTOR
            <= config.DETECTOR_CROP_WIDTH
            and (motion_region["y2"] - motion_region["y1"]) * config.DETECTOR_CROP_PADDING_FACTOR
            <= config.DETECTOR_CROP_HEIGHT
        ):
            x1, y1 = motion_region["x1"], motion_region["y1"]
            x2, y2 = motion_region["x2"], motion_region["y2"]

            cx: int = round((x1 + x2) // 2)
            cy: int = round((y1 + y2) // 2)

            img_h, img_w = img.shape[:2]
            left: int = max(0, cx - config.DETECTOR_CROP_WIDTH // 2)
            right: int = min(img_w, cx + config.DETECTOR_CROP_WIDTH // 2)
            if left == 0:
                right = config.DETECTOR_CROP_WIDTH
            if right == img_w:
                left = img_w - config.DETECTOR_CROP_WIDTH
            top: int = max(0, cy - config.DETECTOR_CROP_HEIGHT // 2)
            bottom: int = min(img_h, cy + config.DETECTOR_CROP_HEIGHT // 2)
            if top == 0:
                bottom = config.DETECTOR_CROP_HEIGHT
            if bottom == img_h:
                top = img_h - config.DETECTOR_CROP_HEIGHT
            img = img[top:bottom, left:right]
            x_offset = left
            y_offset = top
            logger.debug(
                f"Image cropped to (x1, y1) = ({left}, {top}), "
                f"(x2,y2) = ({right}, {bottom}). offset: ({x_offset}, {y_offset})"
            )

        return img, x_offset, y_offset

    def detect(self, motion_region: dict[str, int] | None = None, frame: np.ndarray | None = None) -> Target | None:
        """
        Crop the provided frame around the motion region and send it to the detection API.

        Args:
            motion_region: A dictionary containing the 'x1', 'y1', 'x2', 'y2', and 'count' of the detected motion.
            frame: The full BGR numpy array frame to analyze.

        Returns:
            A Target object containing the bounding box and absolute coordinates, or None if no target is found.
        """
        tA: float = time.time()

        if frame is None:
            logger.error("No frame provided to target detector.")
            return None

        img, x_offset, y_offset = self._prepare_image(frame, motion_region)

        try:
            _, jpeg_buffer = cv2.imencode(".jpg", img)
            res: requests.Response = self.http_session.post(
                config.DETECTOR_API_URL + f"detect/{config.DETECTOR_TARGET_CLASSES}",
                files={"image": ("image.jpg", jpeg_buffer.tobytes(), "image/jpeg")},
                timeout=3,
            )
        except requests.exceptions.Timeout:
            logger.warning("Target detection timeout.")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("Connection Error.")
            return None

        if res.status_code != 200:
            raise ValueError(f"Unexpected response from target detection backend: {res.status_code}: {res.text}")  # noqa: TRY003

        logger.debug(f"Target detection response received after {(time.time() - tA) * 1000:.2f} ms:")
        resj: dict = res.json()
        logger.debug(resj)

        if not resj["success"] or "detections" not in resj:
            logger.warning(f"Target detection failed: {resj}")
            return None

        if len(resj["detections"]) == 0:
            logger.debug("No target detected.")
            return None

        # we will only act on the first detection
        det: dict = resj["detections"][0]
        x1, x2 = det["x1"], det["x2"]
        y1, y2 = det["y1"], det["y2"]

        x1_abs, x2_abs = int(x1 + x_offset), int(x2 + x_offset)
        y1_abs, y2_abs = int(y1 + y_offset), int(y2 + y_offset)

        logger.info(f"Detection of {det['class']} at x1={x1_abs}, y1={y1_abs}, x2={x2_abs}, y2={y2_abs}.")
        return Target(x1=x1_abs, y1=y1_abs, x2=x2_abs, y2=y2_abs)
