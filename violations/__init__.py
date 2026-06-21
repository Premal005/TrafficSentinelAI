# violations/__init__.py
"""Traffic violation detection modules — one specialized expert per violation type."""

from violations.base import BaseViolationDetector, get_registered_detectors
from violations.helmet import HelmetViolationDetector
from violations.seatbelt import SeatbeltViolationDetector
from violations.triple_riding import TripleRidingViolationDetector
from violations.wrong_side import WrongSideViolationDetector
from violations.stop_line import StopLineViolationDetector
from violations.red_light import RedLightViolationDetector
from violations.illegal_parking import IllegalParkingViolationDetector

__all__ = [
    "BaseViolationDetector",
    "get_registered_detectors",
    "HelmetViolationDetector",
    "SeatbeltViolationDetector",
    "TripleRidingViolationDetector",
    "WrongSideViolationDetector",
    "StopLineViolationDetector",
    "RedLightViolationDetector",
    "IllegalParkingViolationDetector",
]

