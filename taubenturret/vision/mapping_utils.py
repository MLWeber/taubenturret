"""Mathematical utilities for translating 2D image coordinates into 3D physical angles."""

import math

from taubenturret import config


def coords_to_angle(x: float, y: float, res: tuple[int, int]) -> tuple[float, float]:
    """
    Convert pixel coordinates from the camera image to physical pan and tilt angles.

    Args:
        x: The x-coordinate in the image.
        y: The y-coordinate in the image.
        res: The resolution of the image (width, height) as a tuple.

    Returns:
        A tuple containing (pan_angle, tilt_angle) in degrees relative to the camera's center.
    """
    f = config.CAM_FOCAL_LENGTH * 1000 / config.CAM_PIXEL_SIZE * config.CAM_SCALE_FACTOR
    cx = (res[0] - 1) / 2
    cy = (res[1] - 1) / 2

    v_x = (x - cx) / f
    v_y = (y - cy) / f

    phi = math.atan(v_x) * 180 / math.pi
    theta = -math.atan(v_y) * 180 / math.pi + config.CAM_TILT

    return phi, theta
