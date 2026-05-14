#!/usr/bin/env python3
"""Main application entry point for the TaubenTurret system."""

import contextlib
import logging
import time

from taubenturret import config
from taubenturret.hardware.turret_controller import TurretController
from taubenturret.vision.camera_controller import CameraController
from taubenturret.vision.mapping_utils import coords_to_angle
from taubenturret.vision.target_detector import TargetDetector
from taubenturret.vision.video_pipeline import VideoPipeline
from taubenturret.web.webserver import start_webserver

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("taubenturret.log"),
        # logging.StreamHandler()
    ],
)


def main() -> None:  # noqa: C901
    """
    Initialize the hardware, configuration, and start the main turret control loop.

    This loop continuously monitors the configured camera_controller for motion.
    When motion is detected, it forwards frames to the TargetDetector, and if a target
    (e.g., a pigeon) is found, it commands the TurretController to aim and fire.
    """
    web_server = None
    turret_controller = None
    video_pipeline = None
    camera_controller = None

    try:
        target_detector: TargetDetector = TargetDetector()
        turret_controller = TurretController()

        camera_controller = CameraController()
        video_pipeline = VideoPipeline(camera_controller)

        manual_fire_requests: list[tuple[float, float]] = []

        def handle_manual_fire(pct_x: float, pct_y: float) -> None:
            manual_fire_requests.append((pct_x, pct_y))

        if config.WEBSERVER_ENABLED:
            try:
                web_server = start_webserver(video_pipeline, fire_callback=handle_manual_fire)
            except Exception:
                logger.exception("Failed to start webserver.")

        logger.info("Starting main loop.")

        t_lastwarn: float = 0
        t_lastmotion: float = 0
        t_lastfire: float = 0
        t_target: float | None = None
        while True:
            t_loopstart: float = time.time()

            try:
                if manual_fire_requests:
                    pct_x, pct_y = manual_fire_requests.pop(0)
                    abs_x = pct_x * config.CAM_RES[0]
                    abs_y = pct_y * config.CAM_RES[1]
                    t_target = t_loopstart
                    t_lastfire = t_loopstart
                    logger.info(f"Manual fire triggered at {pct_x:.2f}, {pct_y:.2f}")
                    turret_controller.fire_at(*coords_to_angle(abs_x, abs_y, config.CAM_RES))

                if video_pipeline.has_motion():
                    if t_loopstart - t_lastmotion > config.TURRET_DEACTIVATION_DELAY:
                        logger.debug("Motion detected.")

                    t_lastmotion = t_loopstart
                    frame, region = video_pipeline.get_current_state()

                    target = None
                    if frame is not None:
                        target = target_detector.detect(region, frame=frame)

                    if target:
                        # bird has been found, fire at it
                        t_target = t_loopstart
                        turret_controller.fire_at(*coords_to_angle(*target.center, config.CAM_RES))
                        t_lastfire = t_loopstart
                        video_pipeline.set_target(target.bbox)
                    else:
                        video_pipeline.set_target(None)

                if t_target and t_loopstart - t_target > config.TURRET_DEACTIVATION_DELAY:
                    logger.debug("Target lost or motion ended. Parking...")
                    turret_controller.park(detach=config.TURRET_PARK_DETACH)
                    video_pipeline.set_target(None)
                    t_target = None

                if (
                    not t_target  # Ensure we don't flush while actively tracking a bird
                    and config.WG_FLUSH_FIRE_INTERVAL > 0
                    and t_loopstart - t_lastfire > config.WG_FLUSH_FIRE_INTERVAL
                    and t_loopstart - t_lastmotion > config.WG_FLUSH_MOTIONLESS_THRESHOLD
                ):
                    # flush the gun, since idling for too long can cause a build up of air in the chamber
                    logger.info("Flushing watergun...")
                    turret_controller.fire_at(
                        *coords_to_angle(config.WG_FLUSH_FIRE_TARGET_X, config.WG_FLUSH_FIRE_TARGET_Y, config.CAM_RES)
                    )
                    t_lastfire = t_loopstart
            except Exception:
                logger.exception("Unexpected error.")

            t_loopend: float = time.time()
            loop_duration: float = t_loopend - t_loopstart
            target_budget: float = 1.0 / config.TARGET_DETECTION_RATE
            delta_t: float = target_budget - loop_duration

            # Only warn if the loop exceeds the budget by more than 25% due to network/AI delays
            if delta_t < -(target_budget * 0.25) and t_loopend - t_lastwarn > 30:
                logger.warning(
                    f"Target detection is slow ({loop_duration * 1000:.0f}ms). "
                    f"Consider lowering the target detection rate."
                )
                t_lastwarn = time.time()
            time.sleep(max(0, delta_t))
    except KeyboardInterrupt:
        logger.info("Exiting...")
    finally:
        if web_server:
            with contextlib.suppress(Exception):
                logger.info("Shutting down webserver...")
                web_server.should_exit = True
        with contextlib.suppress(Exception):
            if turret_controller:
                turret_controller.close()
        with contextlib.suppress(Exception):
            if video_pipeline:
                video_pipeline.close()
        with contextlib.suppress(Exception):
            if camera_controller:
                camera_controller.close()


if __name__ == "__main__":
    main()
