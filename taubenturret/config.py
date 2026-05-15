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

import logging
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _get_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _get_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return str(val).lower() in ("true", "1", "yes", "y", "t")


def _get_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _get_log_level(key: str, default: int) -> int:
    val = os.environ.get(key)
    if not val:
        return default
    if val.isdigit():
        return int(val)
    return getattr(logging, val.upper(), default)


def _get_res(key: str, default: str) -> tuple[int, int]:
    val = os.environ.get(key, default)
    try:
        parts = val.lower().split("x")
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        parts = default.lower().split("x")
        return int(parts[0].strip()), int(parts[1].strip())


# --- Main Application ---
MOTION_DETECTION_RATE: float = _get_float("MOTION_DETECTION_RATE", 10.0)
TARGET_DETECTION_RATE: float = _get_float("TARGET_DETECTION_RATE", 3.0)
LOG_LEVEL: int = _get_log_level("LOG_LEVEL", logging.INFO)

# --- Camera Image Mapping ---
CAM_FPS: int = _get_int("CAM_FPS", 10)
CAM_TILT: float = _get_float("CAM_TILT", 12.0)
CAM_FOV_X: float = _get_float("CAM_FOV_X", 62.28)
CAM_FOV_Y: float = _get_float("CAM_FOV_Y", 48.83)
CAM_FOCAL_LENGTH: float = _get_float("CAM_FOCAL_LENGTH", 3.04)
CAM_PIXEL_SIZE: float = _get_float("CAM_PIXEL_SIZE", 1.12)
CAM_FULL_RES_X: int = _get_int("CAM_FULL_RES_X", 3280)
CAM_FULL_RES_Y: int = _get_int("CAM_FULL_RES_Y", 2464)
CAM_RES: tuple[int, int] = _get_res("CAM_RES", "1280x960")
CAM_STREAM_RES: tuple[int, int] = _get_res("CAM_STREAM_RES", "640x480")
CAM_MOTION_RES_FACTOR: float = _get_float("CAM_MOTION_RES_FACTOR", 0.5)
CAM_MOTION_MIN_AREA: float = _get_float("CAM_MOTION_MIN_AREA", 0.4)
CAM_MOTION_THRESHOLD: int = _get_int("CAM_MOTION_THRESHOLD", 15)
CAM_MOTION_BG_LEARNING_RATE: float = _get_float("CAM_MOTION_BG_LEARNING_RATE", 0.1)
CAM_MOTION_OFF_DELAY: float = _get_float("CAM_MOTION_OFF_DELAY", 5.0)
CAM_SCALE_FACTOR: float = _get_float("CAM_SCALE_FACTOR", 0.39)

# --- Target Detector ---
DETECTOR_API_URL: str = _get_str("DETECTOR_API_URL", "http://127.0.0.1:8081/v1/").rstrip("/") + "/"
DETECTOR_CROP_PADDING_FACTOR: float = _get_float("DETECTOR_CROP_PADDING_FACTOR", 1.2)
DETECTOR_CROP_WIDTH: int = _get_int("DETECTOR_CROP_WIDTH", 640)
DETECTOR_CROP_HEIGHT: int = _get_int("DETECTOR_CROP_HEIGHT", 480)

# --- Turret & Servos ---
TURRET_DEACTIVATION_DELAY: float = _get_float("TURRET_DEACTIVATION_DELAY", 5.0)
TURRET_PARK_PAN: float = _get_float("TURRET_PARK_PAN", -60.0)
TURRET_PARK_TILT: float = _get_float("TURRET_PARK_TILT", -5.0)
TURRET_PARK_DETACH: bool = _get_bool("TURRET_PARK_DETACH", True)
TURRET_MIN_PAN: float = _get_float("TURRET_MIN_PAN", -60.0)
TURRET_MAX_PAN: float = _get_float("TURRET_MAX_PAN", 60.0)
TURRET_MIN_TILT: float = _get_float("TURRET_MIN_TILT", -30.0)
TURRET_MAX_TILT: float = _get_float("TURRET_MAX_TILT", 35.0)

SERVO_PAN_PIN: int = _get_int("SERVO_PAN_PIN", 18)
SERVO_PAN_MIN_PULSEWIDTH: float = _get_float("SERVO_PAN_MIN_PULSEWIDTH", 0.0005)
SERVO_PAN_MAX_PULSEWIDTH: float = _get_float("SERVO_PAN_MAX_PULSEWIDTH", 0.0025)
SERVO_PAN_MIN_ANGLE: float = _get_float("SERVO_PAN_MIN_ANGLE", 90.0)
SERVO_PAN_MAX_ANGLE: float = _get_float("SERVO_PAN_MAX_ANGLE", -90.0)
SERVO_PAN_MIDPOINT: float = _get_float("SERVO_PAN_MIDPOINT", 0.0)
SERVO_PAN_SPEED: float = _get_float("SERVO_PAN_SPEED", 120.0)

SERVO_TILT_PIN: int = _get_int("SERVO_TILT_PIN", 19)
SERVO_TILT_MIN_PULSEWIDTH: float = _get_float("SERVO_TILT_MIN_PULSEWIDTH", 0.0005)
SERVO_TILT_MAX_PULSEWIDTH: float = _get_float("SERVO_TILT_MAX_PULSEWIDTH", 0.0025)
SERVO_TILT_MIN_ANGLE: float = _get_float("SERVO_TILT_MIN_ANGLE", -90.0)
SERVO_TILT_MAX_ANGLE: float = _get_float("SERVO_TILT_MAX_ANGLE", 90.0)
SERVO_TILT_MIDPOINT: float = _get_float("SERVO_TILT_MIDPOINT", 0.0)
SERVO_TILT_SPEED: float = _get_float("SERVO_TILT_SPEED", 60.0)

# --- Watergun ---
WG_RELAY_GPIO_PIN: int = _get_int("WG_RELAY_GPIO_PIN", 2)
WG_RELAY_HIGH_ACTIVE: bool = _get_bool("WG_RELAY_HIGH_ACTIVE", True)
WG_FIRE_COOLDOWN: float = _get_float("WG_FIRE_COOLDOWN", 1.0)
WG_FIRE_DURATION: float = _get_float("WG_FIRE_DURATION", 0.25)
WG_FLUSH_FIRE_INTERVAL: int = _get_int("WG_FLUSH_FIRE_INTERVAL", 0)
WG_FLUSH_MOTIONLESS_THRESHOLD: int = _get_int("WG_FLUSH_MOTIONLESS_THRESHOLD", 5 * 60)
WG_FLUSH_FIRE_TARGET_X: int = _get_int("WG_FLUSH_FIRE_TARGET_X", 320)
WG_FLUSH_FIRE_TARGET_Y: int = _get_int("WG_FLUSH_FIRE_TARGET_Y", 740)

# --- Recordings ---
RECORD_ON_MOTION: bool = _get_bool("RECORD_ON_MOTION", False)
RECORD_DIRECTORY: str = _get_str("RECORD_DIRECTORY", "recordings")
RECORD_BITRATE: int = _get_int("RECORD_BITRATE", 2000000)

# --- Webserver ---
WEBSERVER_ENABLED: bool = _get_bool("WEBSERVER_ENABLED", False)
WEBSERVER_HOST: str = _get_str("WEBSERVER_HOST", "0.0.0.0")  # noqa: S104
WEBSERVER_PORT: int = _get_int("WEBSERVER_PORT", 8080)
WEBSERVER_USERNAME: str = _get_str("WEBSERVER_USERNAME", "admin")
WEBSERVER_PASSWORD: str = _get_str("WEBSERVER_PASSWORD", "taubenturret")
