"""Motion detection and image extraction from picamera2 streams."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from threading import Thread

import numpy as np
from picamera2 import Picamera2, controls

from taubenturret import config

logger = logging.getLogger(__name__)
logging.getLogger("picamera2").setLevel(logging.WARNING)


class CameraController:
    """Controls the Raspberry Pi camera hardware and pushes raw frames to callbacks."""

    def __init__(self) -> None:
        """Initialize the hardware camera source."""
        if Picamera2 is None or np is None:
            msg = "PiCamera2 dependencies (picamera2, numpy) are not installed."
            raise RuntimeError(msg)

        self.frame_delay = 1.0 / config.MOTION_DETECTION_RATE

        self.picam2: Picamera2 | None = None
        self._initialize_camera()

        self.last_process_time = 0.0

        self.exit = False
        self.failed_captures = 0

        self.frame_callbacks: list[Callable[[np.ndarray, np.ndarray], None]] = []

        self.thread = Thread(target=self._run_loop)
        self.thread.daemon = True
        self.thread.start()

    def _initialize_camera(self) -> None:
        """Initialize or completely rebuild the Picamera2 hardware stack."""
        if self.picam2 is not None:
            logger.info("Tearing down existing camera instance for recovery.")
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                logger.debug("Ignored error while closing dead camera instance.", exc_info=True)

        try:
            self.picam2 = Picamera2()
        except IndexError as e:
            msg = "No camera detected by the OS. Check your ribbon cable and ensure legacy camera support is disabled."
            raise RuntimeError(msg) from e

        # two streams: high-res YUV for recording/AI, low-res YUV for motion detection
        picam_config = self.picam2.create_video_configuration(
            main={
                "size": config.CAM_RES,
                "format": "YUV420",
            },
            lores={
                "size": config.CAM_STREAM_RES,
                "format": "YUV420",
            },
            controls={
                "FrameRate": config.CAM_FPS,
                "ExposureValue": 0.5,  # Slightly overexpose to pull details out of shadows
            },
        )
        self.picam2.configure(picam_config)
        self.picam2.start()

        try:
            if controls is not None:
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except Exception:
            logger.debug("Continuous autofocus not supported by this camera module (likely fixed-focus).")

    def add_frame_callback(self, callback: Callable[[np.ndarray, np.ndarray], None]) -> None:
        """Register a callback to receive the raw high-res YUV frame and low-res gray frame on every capture."""
        self.frame_callbacks.append(callback)

    def _run_loop(self) -> None:
        """Internal wrapper to run the main loop with error handling."""
        logger.info(f"Starting {self.__class__.__name__} loop.")
        while not self.exit:
            try:
                self.run()
            except Exception:
                logger.exception(f"Error in {self.__class__.__name__} loop. Attempting recovery.")
                time.sleep(1.0)
                try:
                    self._initialize_camera()
                except Exception:
                    logger.exception("Failed to recover camera. Retrying in 5s...")
                    time.sleep(4.0)

    def run(self) -> None:
        """Pull a single frame from hardware and broadcast it to the callbacks."""
        req = self.picam2.capture_request()
        if not req:
            self.failed_captures += 1
            if self.failed_captures > 5:
                # Reset the counter and raise so the outer loop catches it and sleeps
                self.failed_captures = 0
                msg = "Camera stream stalled or closed unexpectedly."
                raise RuntimeError(msg)
            time.sleep(0.1)
            return

        self.failed_captures = 0

        current_time = time.time()

        if current_time - self.last_process_time < self.frame_delay:
            req.release()
            return

        self.last_process_time = current_time

        try:
            frame_main_yuv = req.make_array("main")
            frame_lores_yuv = req.make_array("lores")

            for cb in self.frame_callbacks:
                try:
                    cb(frame_main_yuv, frame_lores_yuv)
                except Exception:
                    logger.exception("Error executing frame callback.")
        finally:
            req.release()

    def close(self) -> None:
        """Signal the thread to shut down and release the camera hardware."""
        logger.info(f"Closing {self.__class__.__name__}.")

        self.exit = True
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        self.thread.join(timeout=2.0)
