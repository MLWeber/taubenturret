"""Handles video encoding, thumbnail generation, and MP4 muxing."""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

from taubenturret import config

logger = logging.getLogger(__name__)


class Recorder:
    """Helper class to manage Picamera2 recording state and FFmpeg post-processing."""

    def __init__(self) -> None:
        self.is_recording = False
        self.current_record_path: str | None = None
        self.record_start_time: float = 0.0
        self._mux_threads: list[Thread] = []

        if config.RECORD_ON_MOTION:
            Path(config.RECORD_DIRECTORY).mkdir(parents=True, exist_ok=True)

    def start(self, picam2: Any, frame_bgr: np.ndarray | None) -> None:
        """Start saving the video stream to disk (and dump the pre-record buffer)."""
        if not picam2 or self.is_recording:
            return

        current_time = time.time()
        timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%dT%H-%M-%S")
        filepath = str(Path(config.RECORD_DIRECTORY) / f"{timestamp}.h264.tmp")
        self.current_record_path = filepath

        if frame_bgr is not None:
            thumb_path = filepath.replace(".h264.tmp", ".jpg")
            cv2.imwrite(thumb_path, cv2.resize(frame_bgr, (320, 240)))

        self.record_start_time = current_time
        self.is_recording = True

        picam2.start_encoder(H264Encoder(config.RECORD_BITRATE), FileOutput(filepath))

    def stop(self, picam2: Any) -> None:
        """Stop saving to disk and spawn a background thread to mux the file."""
        if not self.is_recording:
            return

        self.is_recording = False

        try:
            if picam2:
                picam2.stop_encoder()
        except Exception:
            logger.debug("Failed to stop encoder cleanly.", exc_info=True)

        if self.current_record_path:
            t = Thread(target=self._mux_to_mp4, args=(self.current_record_path,), daemon=True)
            self._mux_threads.append(t)
            t.start()
            self.current_record_path = None

        # Clean up finished threads to prevent the list from growing indefinitely
        self._mux_threads = [t for t in self._mux_threads if t.is_alive()]

    def close(self) -> None:
        """Ensure all background muxing tasks finish before exiting."""
        for t in self._mux_threads:
            if t.is_alive():
                logger.info("Waiting for background video muxing to finish...")
                t.join(timeout=10.0)

    def _mux_to_mp4(self, h264_path: str) -> None:
        """Fast-mux a raw .h264 stream into an .mp4 container."""
        h264_file = Path(h264_path)
        if not h264_file.exists():
            return
        mp4_path = h264_path.replace(".h264.tmp", ".mp4")
        mp4_tmp_path = mp4_path + ".tmp"

        logger.info(f"Wrapping {h264_file.name} into MP4...")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "h264",
            "-r",
            str(config.CAM_FPS),
            "-i",
            h264_path,
            "-c:v",
            "copy",
            "-f",
            "mp4",
            mp4_tmp_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # noqa: S603

            Path(mp4_tmp_path).rename(mp4_path)
            h264_file.unlink()
        except FileNotFoundError:
            logger.exception("FFmpeg is not installed.")
            h264_file.rename(h264_path.replace(".tmp", ""))
        except subprocess.CalledProcessError as e:
            logger.exception(f"FFmpeg muxing failed: {e.stderr.decode('utf-8', errors='replace')}")
            h264_file.rename(h264_path.replace(".tmp", ""))
            if Path(mp4_tmp_path).exists():
                Path(mp4_tmp_path).unlink()
