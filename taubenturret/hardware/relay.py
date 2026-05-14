"""Control a relay on a GPIO pin."""

import logging

try:
    import gpiod
    from gpiod.line import Direction, Value
except ImportError:
    gpiod = None
    Direction = None
    Value = None

logger = logging.getLogger(__name__)


class Relay:
    """A class to control a GPIO pin using the modern gpiod library."""

    def __init__(self, pin: int, active_high: bool = True, initial_value: bool = False) -> None:
        if gpiod is None:
            msg = "The gpiod library is not installed."
            raise ImportError(msg)

        self.pin = pin
        self.request: gpiod.LineRequest | None = None
        self._chip: gpiod.Chip | None = None

        # On RPi 5/4, header pins are on gpiochip4. On older models, it's gpiochip0.
        chip_paths = ["/dev/gpiochip4", "/dev/gpiochip0"]
        for path in chip_paths:
            if gpiod.is_gpiochip_device(path):
                try:
                    self._chip = gpiod.Chip(path)
                    break
                except PermissionError:
                    logger.warning(f"Permission denied for {path}. Skipping.")
                except Exception:
                    logger.warning(f"Could not open {path}. Skipping.", exc_info=True)

        if not self._chip:
            msg = (
                "Could not open a GPIO chip. Check permissions (are you in the 'gpio' group?) and device availability."
            )
            raise RuntimeError(msg)

        try:
            self.request = self._chip.request_lines(
                consumer="taubenturret-relay",
                config={
                    self.pin: gpiod.LineSettings(
                        direction=Direction.OUTPUT,
                        output_value=Value.ACTIVE if initial_value else Value.INACTIVE,
                        active_low=not active_high,
                    )
                },
            )
        except Exception:
            logger.exception(f"Failed to request GPIO pin {self.pin} via gpiod.")
            if self._chip:
                self._chip.close()
            raise

    def on(self) -> None:
        """Turn the relay on."""
        if self.request:
            self.request.set_value(self.pin, Value.ACTIVE)

    def off(self) -> None:
        """Turn the relay off."""
        if self.request:
            self.request.set_value(self.pin, Value.INACTIVE)

    def close(self) -> None:
        """Turn off and unexport the GPIO pin."""
        if self.request:
            self.off()
            self.request.close()
        if self._chip:
            self._chip.close()
