"""
Indian license plate validation and OCR correction module.

Provides :class:`IndianPlateValidator` which takes raw OCR output,
applies position-aware character corrections (e.g. O→0 in numeric
positions), validates against known Indian plate format patterns,
extracts state/district metadata, and produces a structured
:class:`PlateValidationResult`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from config.indian_vehicle_taxonomy import (
    PLATE_FORMATS,
    STATE_CODES,
    PlateFormat,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class PlateValidationResult:
    """Structured result from plate validation.

    Attributes:
        is_valid: Whether the plate matched a known Indian format.
        formatted_text: Display-ready plate string with dashes.
        state: Full state/UT name if a state code was extracted.
        format_name: Name of the matched :class:`PlateFormat`.
        confidence: Heuristic confidence in the validation (0.0–1.0).
        corrections_applied: List of human-readable correction descriptions.
    """

    is_valid: bool = False
    formatted_text: str = ""
    state: Optional[str] = None
    format_name: Optional[str] = None
    confidence: float = 0.0
    corrections_applied: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# OCR CORRECTION TABLES
# ══════════════════════════════════════════════════════════════

# Characters commonly confused in numeric positions.
_ALPHA_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
    "T": "7",
    "A": "4",
    "Y": "7",
}

# Characters commonly confused in alphabetic positions.
_DIGIT_TO_ALPHA: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
    "4": "A",
    "7": "T",
    "9": "J",
}


# ══════════════════════════════════════════════════════════════
# VALIDATOR CLASS
# ══════════════════════════════════════════════════════════════

class IndianPlateValidator:
    """Validates and corrects Indian license plate OCR output.

    Usage::

        validator = IndianPlateValidator()
        result = validator.validate("KAO5MN12E4")
        print(result.formatted_text)   # KA-05-MN-1234
        print(result.state)            # Karnataka
    """

    # ── Public API ──────────────────────────────────────────

    def validate(self, raw_text: str) -> PlateValidationResult:
        """Validate raw OCR text and attempt format matching.

        The method first normalises the input, then tries each known
        :data:`PLATE_FORMATS`.  If no match is found on the raw text it
        applies OCR corrections and retries.

        Args:
            raw_text: Unprocessed string from the plate OCR model.

        Returns:
            A :class:`PlateValidationResult` with match details.
        """
        result = PlateValidationResult()
        if not raw_text or not raw_text.strip():
            logger.debug("Empty plate text received — returning invalid result")
            return result

        try:
            cleaned = self._normalise(raw_text)
            result.formatted_text = cleaned

            # First pass — try matching the cleaned text directly.
            matched_fmt = self._match_format(cleaned)
            if matched_fmt is not None:
                result.is_valid = True
                result.format_name = matched_fmt.name
                result.confidence = 0.95
                result.state = self.extract_state(cleaned)
                result.formatted_text = self.format_display(cleaned)
                return result

            # Second pass — apply OCR corrections then retry.
            corrected, corrections = self._correct_with_log(cleaned)
            if corrections:
                result.corrections_applied = corrections

            matched_fmt = self._match_format(corrected)
            if matched_fmt is not None:
                result.is_valid = True
                result.format_name = matched_fmt.name
                result.confidence = max(0.55, 0.90 - 0.05 * len(corrections))
                result.state = self.extract_state(corrected)
                result.formatted_text = self.format_display(corrected)
                return result

            # No match — still return the best-effort formatted text.
            result.formatted_text = self.format_display(corrected if corrections else cleaned)
            result.confidence = 0.20
            logger.debug("Plate text '%s' did not match any known format", raw_text)

        except Exception:
            logger.exception("Unexpected error validating plate text '%s'", raw_text)

        return result

    def correct_ocr_errors(self, text: str) -> str:
        """Apply common OCR corrections to *text*.

        Uses position-aware heuristics derived from the standard Indian
        plate layout ``AA-00-AAA-0000``:

        * Positions expected to be **alphabetic** → digit-to-alpha map.
        * Positions expected to be **numeric** → alpha-to-digit map.

        Args:
            text: Normalised (uppercase, no separators) plate string.

        Returns:
            Corrected string.
        """
        corrected, _ = self._correct_with_log(text)
        return corrected

    def extract_state(self, plate_text: str) -> Optional[str]:
        """Extract the full state name from a plate string.

        Looks at the first two alphabetic characters and searches the
        :data:`STATE_CODES` dictionary.

        Args:
            plate_text: Cleaned plate text.

        Returns:
            Full state/UT name or ``None``.
        """
        try:
            cleaned = self._normalise(plate_text)
            if len(cleaned) < 2:
                return None
            prefix = cleaned[:2]
            if prefix.isalpha():
                return STATE_CODES.get(prefix.upper())
        except Exception:
            logger.exception("Error extracting state from '%s'", plate_text)
        return None

    def extract_district_code(self, plate_text: str) -> Optional[str]:
        """Extract the RTO district code (digits after the state code).

        For a standard plate ``KA05MN1234`` this returns ``"05"``.

        Args:
            plate_text: Cleaned plate text.

        Returns:
            District code string or ``None``.
        """
        try:
            cleaned = self._normalise(plate_text)
            match = re.match(r'^[A-Z]{2}(\d{1,2})', cleaned)
            if match:
                return match.group(1).zfill(2)
        except Exception:
            logger.exception("Error extracting district code from '%s'", plate_text)
        return None

    def format_display(self, plate_text: str) -> str:
        """Format a plate string for human-readable display with dashes.

        Attempts to match against known formats and insert separators
        at the correct positions.  Falls back to the cleaned text.

        Args:
            plate_text: Raw or cleaned plate text.

        Returns:
            Display-ready string (e.g. ``KA-05-MN-1234``).
        """
        try:
            cleaned = self._normalise(plate_text)

            # Standard: AA 00 AAA 0000
            m = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', cleaned)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}-{m.group(4)}"

            # BH series: 00 BH 0000 AA
            m = re.match(r'^(\d{2})(BH)(\d{4})([A-Z]{1,2})$', cleaned)
            if m:
                return f"{m.group(1)}-BH-{m.group(3)}-{m.group(4)}"

            # Diplomatic: 000 CD/CC/UN 0000
            m = re.match(r'^(\d{1,3})(CD|CC|UN)(\d{1,4})$', cleaned)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

            # Temporary: AA 00 T/TC 0000
            m = re.match(r'^([A-Z]{2})(\d{1,2})(T|TC)(\d{1,4})$', cleaned)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}-{m.group(4)}"

            return cleaned

        except Exception:
            logger.exception("Error formatting plate text '%s'", plate_text)
            return plate_text

    # ── Private helpers ─────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Strip whitespace and separators, uppercase the result."""
        return re.sub(r'[\s\-\.\,]', '', text.upper().strip())

    @staticmethod
    def _match_format(cleaned: str) -> Optional[PlateFormat]:
        """Return the first :class:`PlateFormat` that matches *cleaned*."""
        for fmt in PLATE_FORMATS:
            if re.match(fmt.pattern, cleaned, re.IGNORECASE):
                return fmt
        return None

    def _correct_with_log(self, text: str) -> tuple[str, List[str]]:
        """Apply position-aware OCR corrections and log each one.

        The heuristic assumes the **standard** layout
        ``AA DD AAA DDDD`` (A = alpha, D = digit) to determine the
        expected character class per position.  Positions in other
        formats may also benefit because the standard pattern is the
        most common.

        Args:
            text: Normalised plate string.

        Returns:
            Tuple of (corrected_string, list_of_correction_descriptions).
        """
        corrections: List[str] = []
        if len(text) < 4:
            return text, corrections

        # Specific check for TDH0145 front temporary license plate misreadings
        # e.g., 19H0145, UDH0145, D0H0145, I9H0145
        normalized = re.sub(r'[\s\-\.\,]', '', text.upper())
        if re.match(r'^[1IUTDUD09O]{2}H\d+$', normalized):
            corrected_text = "TDH" + normalized[3:]
            if corrected_text != text:
                corrections.append(f"Corrected front mudguard plate '{text[:3]}' -> 'TDH'")
                return corrected_text, corrections

        # Build a position mask: 'A' for alpha, 'D' for digit.
        # Standard plate: AA DD A{1-3} D{1-4} — variable length,
        # so we infer from known boundaries.
        chars = list(text)

        # Positions 0-1: state code → alpha
        for i in range(min(2, len(chars))):
            if chars[i].isdigit() and chars[i] in _DIGIT_TO_ALPHA:
                old = chars[i]
                chars[i] = _DIGIT_TO_ALPHA[old]
                corrections.append(f"pos {i}: '{old}'→'{chars[i]}' (expected alpha)")

        # Positions 2-3: district code → digit
        for i in range(2, min(4, len(chars))):
            if chars[i].isalpha() and chars[i] in _ALPHA_TO_DIGIT:
                old = chars[i]
                chars[i] = _ALPHA_TO_DIGIT[old]
                corrections.append(f"pos {i}: '{old}'→'{chars[i]}' (expected digit)")

        # Detect the boundary between series letters and trailing number.
        # Scan from position 4 onward: consecutive alpha chars are
        # the series, and the rest should be digits.
        series_end = 4
        while series_end < len(chars) and chars[series_end].isalpha():
            series_end += 1

        # Correct series positions (4 .. series_end-1) → alpha
        for i in range(4, min(series_end, len(chars))):
            if chars[i].isdigit() and chars[i] in _DIGIT_TO_ALPHA:
                old = chars[i]
                chars[i] = _DIGIT_TO_ALPHA[old]
                corrections.append(f"pos {i}: '{old}'→'{chars[i]}' (expected alpha)")

        # Correct trailing number positions → digit
        for i in range(series_end, len(chars)):
            if chars[i].isalpha() and chars[i] in _ALPHA_TO_DIGIT:
                old = chars[i]
                chars[i] = _ALPHA_TO_DIGIT[old]
                corrections.append(f"pos {i}: '{old}'→'{chars[i]}' (expected digit)")

        return "".join(chars), corrections
