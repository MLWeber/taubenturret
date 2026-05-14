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

"""Hardware controller for the water pump relay."""

import logging
import time
from threading import Timer

from taubenturret import config
from taubenturret.hardware.relay import Relay

logger = logging.getLogger(__name__)


class WatergunController:
    """Controls the relay triggering the physical watergun mechanism."""

    def __init__(self) -> None:
        """
        Initialize the relay control.
        """
        self.relay = Relay(
            pin=config.WG_RELAY_GPIO_PIN,
            active_high=config.WG_RELAY_HIGH_ACTIVE,
            initial_value=False,
        )
        self.t_last_shot: float = 0.0
        self.timer: Timer | None = None

    def fire(self) -> None:
        """
        Trigger the watergun relay.

        Silently ignores the command if the cooldown interval has not elapsed.
        """
        if time.time() - self.t_last_shot < config.WG_FIRE_COOLDOWN:
            return
        self.t_last_shot = time.time()
        self.relay.on()
        logger.info("Fire!")
        self.timer = Timer(config.WG_FIRE_DURATION, self.relay.off)
        self.timer.start()

    def close(self) -> None:
        """Ensure the relay is safely turned off during shutdown."""
        if self.timer is not None:
            self.timer.cancel()
        self.relay.off()
