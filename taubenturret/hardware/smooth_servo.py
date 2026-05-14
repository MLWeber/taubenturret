"""Provides a thread-safe wrapper around rpi_hardware_pwm to enforce smooth servo movements."""

import logging
import time
from threading import Lock, Thread
from typing import Any

from rpi_hardware_pwm import HardwarePWM

logger = logging.getLogger(__name__)


class SmoothServo:
    """A Servo that smoothly interpolates motion over time using hardware PWM instead of snapping instantly."""

    def __init__(
        self,
        pwm_channel: int,
        min_angle: float = -90.0,
        max_angle: float = 90.0,
        min_pulse_width: float = 0.001,
        max_pulse_width: float = 0.002,
        frame_width: float = 0.02,
        speed: float = 90.0,
        update_interval: float = 0.02,
        tolerance: float = 0.5,
        initial_angle: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the smooth servo.

        Args:
            pwm_channel: The hardware PWM pin (18 or 19).
            min_angle: The minimum angle of the servo.
            max_angle: The maximum angle of the servo.
            min_pulse_width: The pulse width corresponding to the minimum angle (in seconds).
            max_pulse_width: The pulse width corresponding to the maximum angle (in seconds).
            frame_width: The length of a PWM frame (in seconds). Default 0.02s (50Hz).
            speed: The maximum sweeping speed in degrees per second.
            update_interval: How frequently (in seconds) the physical angle should be updated.
            tolerance: Distance to the target in degrees considered "close enough" to stop.
            initial_angle: The initial angle to set the servo to.
            **kwargs: Ignored. Included for backward compatibility.
        """
        channel_request = int(pwm_channel)
        pwm_map = {12: 0, 18: 0, 40: 0, 52: 0, 13: 1, 19: 1, 41: 1, 45: 1, 53: 1}
        if channel_request in pwm_map:
            self.pwm_channel: int = pwm_map[channel_request]
        elif channel_request in [0, 1]:
            self.pwm_channel = channel_request
        else:
            msg = f"BCM pin {channel_request} does not support Hardware PWM."
            raise ValueError(msg)

        self.min_angle: float = min_angle
        self.max_angle: float = max_angle
        self.min_pulse_width: float = min_pulse_width
        self.max_pulse_width: float = max_pulse_width
        self.frame_width: float = frame_width

        self.active: bool = False
        self.speed: float = speed
        self.update_interval: float = update_interval
        self.tolerance: float = tolerance
        self._lock = Lock()

        hz = int(1.0 / self.frame_width)
        self.pwm = HardwarePWM(pwm_channel=self.pwm_channel, hz=hz)
        self.pwm.start(0)  # 0 duty cycle = unpowered/limp
        self._is_limp: bool = True

        self._target_angle: float | None = initial_angle
        self._current_angle: float | None = None
        self.last_known_angle: float | None = initial_angle
        self.max_step_size: float = self.speed * self.update_interval

        if initial_angle is not None:
            self.angle = initial_angle

    @property
    def angle(self) -> float | None:
        """The current physical angle of the servo."""
        return self._current_angle

    @angle.setter
    def angle(self, new_angle: float | None) -> None:
        """Set the physical angle of the servo, updating the hardware PWM."""
        if new_angle is None:
            self.detach()
            return

        # Safely clamp the angle whether min_angle < max_angle or min_angle > max_angle
        lower_bound = min(self.min_angle, self.max_angle)
        upper_bound = max(self.min_angle, self.max_angle)
        new_angle = max(lower_bound, min(upper_bound, new_angle))

        angle_range = self.max_angle - self.min_angle
        pulse_width_range = self.max_pulse_width - self.min_pulse_width

        if angle_range != 0:
            pulse_width = self.min_pulse_width + (new_angle - self.min_angle) * pulse_width_range / angle_range
        else:
            pulse_width = self.min_pulse_width

        duty_cycle = (pulse_width / self.frame_width) * 100.0

        if self._is_limp:
            self._is_limp = False

        self.pwm.change_duty_cycle(duty_cycle)
        self._current_angle = new_angle

    def detach(self) -> None:
        """Stop sending the PWM signal, putting the servo in a limp/unpowered state."""
        with self._lock:
            self._target_angle = None
        self.pwm.change_duty_cycle(0)
        self._is_limp = True
        self._current_angle = None

    @property
    def target_angle(self) -> float | None:
        """The current destination angle the servo is moving toward."""
        with self._lock:
            return self._target_angle

    @target_angle.setter
    def target_angle(self, angle: float | None) -> None:
        """Set a new destination angle, starting the background interpolation thread if needed."""
        with self._lock:
            self._target_angle = angle
            if self.active:
                return
            self.active = True
            Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self) -> None:
        """Background loop executing the continuous software interpolation toward the target."""
        while True:
            tA = time.time()

            with self._lock:
                target = self._target_angle

            try:
                if target is None:
                    self.angle = None
                elif self.last_known_angle is None:
                    self.angle = target
                    self.last_known_angle = target
                else:
                    delta = target - self.last_known_angle
                    if abs(delta) > self.max_step_size:
                        delta = self.max_step_size if delta > 0 else -self.max_step_size

                    new_angle = self.last_known_angle + delta
                    self.angle = new_angle
                    self.last_known_angle = new_angle

            except Exception:
                logger.exception("Failed to update servo.")

            with self._lock:
                if self._target_angle == target and (
                    target is None
                    or (self.last_known_angle is not None and abs(target - self.last_known_angle) <= self.tolerance)
                ):
                    self.active = False
                    break

            dt = self.update_interval - (time.time() - tA)
            if dt > 0:
                time.sleep(dt)

    def close(self) -> None:
        """Cleanly stop the hardware PWM."""
        self.pwm.stop()
