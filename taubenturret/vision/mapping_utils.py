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
