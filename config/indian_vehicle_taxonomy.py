"""
Indian Vehicle Taxonomy & License Plate Formats.

Specialized knowledge module for Indian traffic context:
- Vehicle categories specific to Indian roads
- License plate format patterns (standard, BH-series, old format)
- State code lookup table (RTO codes)
- Plate color classification (private/commercial/EV/diplomatic)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple


# ══════════════════════════════════════════════════════════════
# INDIAN VEHICLE CATEGORIES
# ══════════════════════════════════════════════════════════════

class VehicleCategory(Enum):
    """Vehicle categories specific to Indian roads."""
    TWO_WHEELER = "two_wheeler"           # Motorcycle, scooter, moped
    THREE_WHEELER = "three_wheeler"       # Auto-rickshaw, e-rickshaw
    FOUR_WHEELER_PRIVATE = "four_wheeler_private"   # Cars, SUVs
    FOUR_WHEELER_COMMERCIAL = "four_wheeler_commercial"  # Taxis, cabs
    HEAVY_VEHICLE = "heavy_vehicle"       # Trucks, trailers
    PUBLIC_TRANSPORT = "public_transport" # Buses (BMTC, KSRTC)
    BICYCLE = "bicycle"
    EMERGENCY = "emergency"              # Ambulance, fire, police
    GOVERNMENT = "government"


class PlateColor(Enum):
    """Indian license plate color codes."""
    WHITE = "white"           # Private vehicles
    YELLOW = "yellow"         # Commercial vehicles (taxis, autos)
    GREEN = "green"           # Electric vehicles
    BLUE = "blue"             # Diplomatic / foreign missions
    RED = "red"               # President / Governor vehicles
    BLACK = "black"           # Self-drive rental vehicles
    ARROW_MARK = "arrow_mark" # Military vehicles


# ══════════════════════════════════════════════════════════════
# STATE RTO CODES
# ══════════════════════════════════════════════════════════════

STATE_CODES: Dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TD": "Temporary Registration",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}


# ══════════════════════════════════════════════════════════════
# LICENSE PLATE FORMAT PATTERNS
# ══════════════════════════════════════════════════════════════

@dataclass
class PlateFormat:
    """Indian license plate format definition."""
    name: str
    pattern: str  # Regex pattern
    example: str
    description: str
    
    def match(self, text: str) -> Optional[re.Match]:
        """Try to match cleaned text against this format."""
        cleaned = self._clean_plate_text(text)
        return re.match(self.pattern, cleaned, re.IGNORECASE)
    
    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """Normalize OCR output for plate matching."""
        # Remove spaces, hyphens, dots
        cleaned = re.sub(r'[\s\-\.\,]', '', text.upper().strip())
        # Common OCR corrections
        corrections = {
            'O': '0', 'I': '1', 'S': '5', 'B': '8',
            'G': '6', 'Z': '2', 'Q': '0',
        }
        # Only apply corrections in numeric positions
        # (This is handled by the validator, not here)
        return cleaned


# Standard Indian plate formats
PLATE_FORMATS: List[PlateFormat] = [
    PlateFormat(
        name="Standard Format",
        pattern=r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$',
        example="KA-05-MN-1234",
        description="State(2) + District(1-2) + Series(1-3) + Number(1-4)"
    ),
    PlateFormat(
        name="BH Series (National)",
        pattern=r'^(\d{2})(BH)(\d{4})([A-Z]{1,2})$',
        example="22-BH-1234-AB",
        description="Year(2) + BH + Number(4) + Series(1-2)"
    ),
    PlateFormat(
        name="Old Format (Single Series)",
        pattern=r'^([A-Z]{2})(\d{1,2})([A-Z])(\d{1,4})$',
        example="KA-01-A-1234",
        description="State(2) + District(1-2) + Series(1) + Number(1-4)"
    ),
    PlateFormat(
        name="Diplomatic",
        pattern=r'^(\d{1,3})(CD|CC|UN)(\d{1,4})$',
        example="24-CD-1234",
        description="Country(1-3) + Type(CD/CC/UN) + Number(1-4)"
    ),
    PlateFormat(
        name="Temporary",
        pattern=r'^([A-Z]{2})(\d{1,2})(T|TC)(\d{1,4})$',
        example="KA-01-TC-1234",
        description="State(2) + District(1-2) + T/TC + Number(1-4)"
    ),
    PlateFormat(
        name="Custom/Temporary Format",
        pattern=r'^([A-Z]{3,4})(\d{1,4})$',
        example="TDH-0145",
        description="Prefix(3-4) + Number(1-4)"
    ),
]


class IndianVehicleTaxonomy:
    """
    Indian traffic context knowledge base.
    
    Provides methods to:
    - Validate and parse Indian license plates
    - Classify plate colors
    - Look up state/RTO information
    - Map detected vehicles to Indian categories
    """
    
    @staticmethod
    def validate_plate(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate text as an Indian license plate.
        
        Returns:
            (is_valid, format_name, state_name)
        """
        for fmt in PLATE_FORMATS:
            match = fmt.match(text)
            if match:
                # Try to extract state code
                state_code = match.group(1) if len(match.group(1)) == 2 and match.group(1).isalpha() else None
                state_name = STATE_CODES.get(state_code) if state_code else None
                return True, fmt.name, state_name
        return False, None, None
    
    @staticmethod
    def get_state_from_code(code: str) -> Optional[str]:
        """Get state name from RTO code."""
        return STATE_CODES.get(code.upper())
    
    @staticmethod
    def format_plate_number(raw_text: str) -> str:
        """
        Format raw OCR text into standard Indian plate format.
        e.g., 'KA05MN1234' → 'KA-05-MN-1234'
        """
        cleaned = re.sub(r'[\s\-\.\,]', '', raw_text.upper().strip())
        
        # Try standard format: XX-00-XX-0000
        match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', cleaned)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}-{match.group(4)}"
        
        # Try BH format: 00-BH-0000-XX
        match = re.match(r'^(\d{2})(BH)(\d{4})([A-Z]{1,2})$', cleaned)
        if match:
            return f"{match.group(1)}-BH-{match.group(3)}-{match.group(4)}"
        
        # Return cleaned if no format matches
        return cleaned
    
    @staticmethod
    def classify_plate_color_hsv(h: float, s: float, v: float) -> PlateColor:
        """
        Classify plate color from average HSV values of plate region.
        
        Args:
            h: Hue (0-180 in OpenCV)
            s: Saturation (0-255)
            v: Value/Brightness (0-255)
        """
        # White plate (private) — low saturation, high value
        if s < 50 and v > 150:
            return PlateColor.WHITE
        
        # Yellow plate (commercial) — hue 20-35, high saturation
        if 15 <= h <= 35 and s > 100 and v > 100:
            return PlateColor.YELLOW
        
        # Green plate (EV) — hue 35-85
        if 35 <= h <= 85 and s > 50:
            return PlateColor.GREEN
        
        # Blue plate (diplomatic) — hue 100-130
        if 100 <= h <= 130 and s > 50:
            return PlateColor.BLUE
        
        # Red plate (VIP) — hue 0-10 or 170-180
        if (h <= 10 or h >= 170) and s > 100:
            return PlateColor.RED
        
        # Default to white
        return PlateColor.WHITE
