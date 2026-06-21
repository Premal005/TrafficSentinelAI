"""
Abstract base class for traffic violation detectors.

Defines the :class:`BaseViolationDetector` contract that every concrete
violation module (``helmet.py``, ``seatbelt.py``, ``triple_riding.py``,
etc.) must implement.  The base class also provides:

* An ``enabled`` toggle to turn detectors on/off at runtime.
* A ``min_confidence`` threshold below which detections are suppressed.
* A lightweight auto-registration pattern so the pipeline can discover
  all installed detector subclasses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Type, TYPE_CHECKING

import numpy as np

from config.settings import ViolationType, SETTINGS

if TYPE_CHECKING:
    from core.scene_graph import SceneGraph
    from core.violation_engine import ViolationRecord

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# DETECTOR REGISTRY
# ══════════════════════════════════════════════════════════════

# Global registry populated by __init_subclass__.
_DETECTOR_REGISTRY: Dict[ViolationType, Type["BaseViolationDetector"]] = {}


def get_registered_detectors() -> Dict[ViolationType, Type["BaseViolationDetector"]]:
    """Return a copy of the detector registry.

    Returns:
        Mapping of :class:`ViolationType` → detector subclass.
    """
    return dict(_DETECTOR_REGISTRY)


# ══════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ══════════════════════════════════════════════════════════════

class BaseViolationDetector(ABC):
    """Interface for a single-violation-type detector.

    Every concrete detector declares *which* violation type it handles
    via the :attr:`violation_type` class attribute and implements the
    :meth:`detect` method.

    Subclasses are automatically registered in the global detector
    registry on class definition (via ``__init_subclass__``).

    Example::

        class HelmetViolationDetector(BaseViolationDetector):
            violation_type = ViolationType.HELMET_NON_COMPLIANCE

            def detect(self, scene_graph, frame=None):
                ...

    Attributes:
        violation_type: The :class:`ViolationType` this detector handles.
            **Must** be overridden in every concrete subclass.
        enabled: Runtime toggle — when ``False`` the detector's
            :meth:`detect` call is short-circuited.
        min_confidence: Detections below this threshold are silently
            discarded by :meth:`run`.
    """

    # ── Class-level attributes (overridden by subclasses) ──────

    violation_type: ClassVar[ViolationType]

    # ── Instance defaults ──────────────────────────────────────

    def __init__(
        self,
        enabled: bool = True,
        min_confidence: Optional[float] = None,
    ) -> None:
        """Initialise the detector.

        Args:
            enabled: Whether this detector is active.
            min_confidence: Floor confidence for reported violations.
                Defaults to :pyattr:`SETTINGS.min_violation_confidence`.
        """
        self.enabled: bool = enabled
        self.min_confidence: float = (
            min_confidence
            if min_confidence is not None
            else SETTINGS.min_violation_confidence
        )
        logger.info(
            "Initialised %s (enabled=%s, min_conf=%.2f)",
            self.__class__.__name__,
            self.enabled,
            self.min_confidence,
        )

    # ── Auto-registration hook ─────────────────────────────────

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register concrete subclasses that define ``violation_type``."""
        super().__init_subclass__(**kwargs)
        vtype = getattr(cls, "violation_type", None)
        if vtype is not None and isinstance(vtype, ViolationType):
            if vtype in _DETECTOR_REGISTRY:
                logger.warning(
                    "Overwriting detector for %s: %s → %s",
                    vtype.value,
                    _DETECTOR_REGISTRY[vtype].__name__,
                    cls.__name__,
                )
            _DETECTOR_REGISTRY[vtype] = cls
            logger.debug("Registered detector %s → %s", vtype.value, cls.__name__)

    # ── Abstract interface ─────────────────────────────────────

    @abstractmethod
    def detect(
        self,
        scene_graph: "SceneGraph",
        frame: Optional[np.ndarray] = None,
    ) -> List["ViolationRecord"]:
        """Run violation detection on a scene graph.

        Subclasses must implement this method.  The scene graph already
        contains entity nodes (vehicles, persons, safety gear, etc.)
        and relationship edges.  The raw *frame* is optionally provided
        for visual-feature based heuristics.

        Args:
            scene_graph: Structured scene representation for the
                current frame.
            frame: Optional BGR image of the current frame — useful for
                detectors that need pixel-level analysis (e.g. seatbelt
                visibility checks).

        Returns:
            A list of :class:`ViolationRecord` instances.  May be empty
            if no violation is detected.
        """
        ...

    # ── Pipeline-facing entry point ────────────────────────────

    def run(
        self,
        scene_graph: "SceneGraph",
        frame: Optional[np.ndarray] = None,
    ) -> List["ViolationRecord"]:
        """Execute the detector with guard checks.

        This is the method the pipeline calls.  It checks the
        ``enabled`` flag, delegates to :meth:`detect`, filters by
        ``min_confidence``, and catches unexpected exceptions so that
        one broken detector cannot crash the full pipeline.

        Args:
            scene_graph: Structured scene graph.
            frame: Optional raw frame.

        Returns:
            Filtered list of :class:`ViolationRecord` instances.
        """
        if not self.enabled:
            return []

        try:
            raw_violations = self.detect(scene_graph, frame)
        except Exception:
            logger.exception(
                "%s raised an unexpected error during detect()",
                self.__class__.__name__,
            )
            return []

        # Filter by minimum confidence
        filtered: List["ViolationRecord"] = []
        for v in raw_violations:
            try:
                if v.confidence >= self.min_confidence:
                    filtered.append(v)
                else:
                    logger.debug(
                        "%s: discarding violation (conf=%.2f < min=%.2f)",
                        self.__class__.__name__,
                        v.confidence,
                        self.min_confidence,
                    )
            except AttributeError:
                # ViolationRecord may not have confidence — keep it.
                filtered.append(v)

        logger.info(
            "%s: %d violations detected (%d after confidence filter)",
            self.__class__.__name__,
            len(raw_violations),
            len(filtered),
        )
        return filtered

    # ── Convenience ────────────────────────────────────────────

    def enable(self) -> None:
        """Enable this detector."""
        self.enabled = True
        logger.info("%s enabled", self.__class__.__name__)

    def disable(self) -> None:
        """Disable this detector."""
        self.enabled = False
        logger.info("%s disabled", self.__class__.__name__)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"type={self.violation_type.value} "
            f"enabled={self.enabled} "
            f"min_conf={self.min_confidence:.2f}>"
        )
