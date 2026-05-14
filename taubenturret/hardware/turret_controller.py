"""Orchestrates the hardware components (servos and watergun) for targeting and firing."""

import logging
from threading import Timer

from taubenturret import config
from taubenturret.hardware.smooth_servo import SmoothServo
from taubenturret.hardware.watergun_controller import WatergunController

logger = logging.getLogger(__name__)


class TurretController:
    """High-level controller for aiming the pan/tilt servos and firing the watergun."""

    def __init__(self) -> None:
        """Initialize the turret's watergun and servos, and move to the parking position."""
        self.watergun: WatergunController = WatergunController()

        self.firing_timer: Timer | None = None
        self.release_timer: Timer | None = None
        self.exit: bool = False

        self.servo_pan = SmoothServo(
            pwm_channel=config.SERVO_PAN_PIN,
            speed=config.SERVO_PAN_SPEED,
            min_pulse_width=config.SERVO_PAN_MIN_PULSEWIDTH,
            max_pulse_width=config.SERVO_PAN_MAX_PULSEWIDTH,
            min_angle=config.SERVO_PAN_MIN_ANGLE + config.SERVO_PAN_MIDPOINT,
            max_angle=config.SERVO_PAN_MAX_ANGLE + config.SERVO_PAN_MIDPOINT,
            initial_angle=config.TURRET_PARK_PAN,
        )
        self.servo_tilt = SmoothServo(
            pwm_channel=config.SERVO_TILT_PIN,
            speed=config.SERVO_TILT_SPEED,
            min_pulse_width=config.SERVO_TILT_MIN_PULSEWIDTH,
            max_pulse_width=config.SERVO_TILT_MAX_PULSEWIDTH,
            min_angle=config.SERVO_TILT_MIN_ANGLE + config.SERVO_TILT_MIDPOINT,
            max_angle=config.SERVO_TILT_MAX_ANGLE + config.SERVO_TILT_MIDPOINT,
            initial_angle=config.TURRET_PARK_TILT,
        )

        self.park()

    def fire_at(self, phi: float, theta: float) -> None:
        """
        Aim the servos at the specified angles and schedule a watergun shot.

        Args:
            phi: The pan angle in degrees.
            thet: The tilt angle in degrees.
        """
        if self.release_timer:
            self.release_timer.cancel()
        if self.firing_timer:
            self.firing_timer.cancel()

        t_wait = self.aim_at(phi, theta)

        self.firing_timer = Timer(t_wait, self.watergun.fire)
        self.firing_timer.start()

    def aim_at(self, phi: float, theta: float) -> float:
        """
        Move the turret to the specified pan and tilt angles.

        Args:
            phi: The pan angle in degrees.
            theta: The tilt angle in degrees.

        Returns:
            The estimated time in seconds required for the servos to complete the physical travel.
        """
        phi = max(min(phi, config.TURRET_MAX_PAN), config.TURRET_MIN_PAN)
        theta = max(min(theta, config.TURRET_MAX_TILT), config.TURRET_MIN_TILT)
        logger.debug(f"Targeting phi={phi:.2f}°, theta={theta:.2f}°")

        delta_phi = phi - self.servo_pan.angle if self.servo_pan.angle is not None else phi - config.TURRET_PARK_PAN
        delta_theta = (
            theta - self.servo_tilt.angle if self.servo_tilt.angle is not None else theta - config.TURRET_PARK_TILT
        )
        t_travel = max(abs(delta_phi) / config.SERVO_PAN_SPEED, abs(delta_theta) / config.SERVO_TILT_SPEED) + 0.25

        self.servo_pan.target_angle = phi
        self.servo_tilt.target_angle = theta

        return t_travel

    def park(self, detach: bool = False) -> None:
        """
        Return the turret to its default resting angles.

        Args:
            detach: If True, physically disable the PWM signals to the servos after parking.
        """
        if self.release_timer:
            self.release_timer.cancel()
        if self.firing_timer:
            self.firing_timer.cancel()
        self.servo_pan.target_angle = config.TURRET_PARK_PAN
        self.servo_tilt.target_angle = config.TURRET_PARK_TILT
        if detach:
            self.release_timer = Timer(5, self.release_servos)
            self.release_timer.start()
        logger.debug("Turret parked.")

    def release_servos(self) -> None:
        """Stop sending PWM signals to the pan and tilt servos to save power and prevent jitter."""
        if self.release_timer:
            self.release_timer.cancel()
        self.servo_pan.detach()
        self.servo_tilt.detach()
        logger.debug("Servos released.")

    def close(self) -> None:
        """Cleanly shut down the turret controller, park the servos, and release hardware resources."""
        self.exit = True
        self.release_servos()
        self.watergun.close()
        self.servo_pan.close()
        self.servo_tilt.close()
