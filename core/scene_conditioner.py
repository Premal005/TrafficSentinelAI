"""
Layer 1: Adaptive Scene Conditioner

Differentiator #2 — Instead of applying fixed preprocessing, we first
classify the image degradation type (low-light, rain, blur, shadow, glare)
and then apply targeted enhancement.

Pipeline:
    Input Image → Degradation Classifier → Targeted Enhancement → Quality Score

Most competitors will use no preprocessing or fixed CLAHE. Our adaptive
approach shows robustness to real-world Bengaluru conditions (monsoon rain,
night, dust, glare from headlights).
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

from config.settings import Settings, SETTINGS, DegradationType, PreprocessorConfig

logger = logging.getLogger(__name__)


@dataclass
class ConditioningResult:
    """Result of adaptive scene conditioning."""
    enhanced_image: np.ndarray
    degradation_type: DegradationType
    quality_score: float  # 0.0 (worst) to 1.0 (best)
    brightness: float
    contrast: float
    sharpness: float
    enhancements_applied: list


class SceneConditioner:
    """
    Adaptive Scene Conditioner (Layer 1 of HSUP).
    
    Detects image degradation type and applies targeted enhancement
    to maximize downstream detection accuracy.
    
    Supported degradation types:
        - CLEAR: Minimal processing (normalization only)
        - LOW_LIGHT: Adaptive gamma correction + CLAHE
        - RAIN: Guided filter dehazing + contrast boost
        - SHADOW: Retinex-based shadow equalization
        - BLUR: Unsharp masking + edge enhancement
        - GLARE: Flare suppression + local contrast normalization
    
    Usage:
        conditioner = SceneConditioner()
        result = conditioner.condition(image)
        enhanced = result.enhanced_image
    """
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or SETTINGS
        self.config: PreprocessorConfig = self.settings.preprocessor
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        logger.info("SceneConditioner initialized with adaptive preprocessing")
    
    def condition(self, image: np.ndarray) -> ConditioningResult:
        """
        Run full adaptive conditioning pipeline.
        
        Args:
            image: Input BGR image (OpenCV format)
            
        Returns:
            ConditioningResult with enhanced image and metadata
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        # Step 1: Analyze image characteristics
        brightness, contrast, sharpness = self._compute_image_stats(image)
        
        # Step 2: Classify degradation type
        degradation = self._classify_degradation(image, brightness, contrast, sharpness)
        logger.info(f"Detected degradation: {degradation.value} "
                    f"(brightness={brightness:.1f}, contrast={contrast:.1f}, sharpness={sharpness:.1f})")
        
        # Step 3: Apply targeted enhancement
        enhanced, enhancements = self._apply_enhancement(image, degradation)
        
        # Step 4: Final normalization
        enhanced = self._normalize(enhanced)
        
        # Step 5: Compute quality score of enhanced image
        b_new, c_new, s_new = self._compute_image_stats(enhanced)
        quality_score = self._compute_quality_score(b_new, c_new, s_new)
        
        return ConditioningResult(
            enhanced_image=enhanced,
            degradation_type=degradation,
            quality_score=quality_score,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            enhancements_applied=enhancements,
        )
    
    # ──────────────────────────────────────────────
    # IMAGE ANALYSIS
    # ──────────────────────────────────────────────
    
    def _compute_image_stats(self, image: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute brightness, contrast, and sharpness metrics.
        
        Args:
            image: BGR image
            
        Returns:
            (brightness, contrast, sharpness) — higher is better for each
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness: mean pixel value (0-255)
        brightness = float(np.mean(gray))
        
        # Contrast: standard deviation of pixel values
        contrast = float(np.std(gray))
        
        # Sharpness: Laplacian variance (higher = sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        
        return brightness, contrast, sharpness
    
    def _classify_degradation(
        self,
        image: np.ndarray,
        brightness: float,
        contrast: float,
        sharpness: float,
    ) -> DegradationType:
        """
        Classify the dominant degradation type of the image.
        
        Uses a rule-based classifier analyzing:
        - Brightness histogram for low-light/glare
        - Laplacian variance for blur
        - Color channel analysis for rain/haze
        - Shadow detection via illumination map
        
        Args:
            image: BGR image
            brightness: Mean brightness
            contrast: Std of brightness
            sharpness: Laplacian variance
        """
        # Check for low-light
        if brightness < self.config.low_light_threshold:
            return DegradationType.LOW_LIGHT
        
        # Check for glare / overexposure
        if brightness > self.config.glare_threshold:
            return DegradationType.GLARE
        
        # Check for blur (motion or defocus)
        if sharpness < self.config.blur_threshold:
            return DegradationType.BLUR
        
        # Check for rain / haze (low contrast + specific color properties)
        if self._detect_rain_haze(image, contrast):
            return DegradationType.RAIN
        
        # Check for strong shadows
        if self._detect_shadows(image):
            return DegradationType.SHADOW
        
        return DegradationType.CLEAR
    
    def _detect_rain_haze(self, image: np.ndarray, contrast: float) -> bool:
        """
        Detect rain or haze conditions.
        
        Rain/haze images typically have:
        - Low contrast (washed out)
        - Shifted color histogram towards white/gray
        - Reduced saturation
        """
        if contrast > 50:  # Reasonable contrast, probably not rain/haze
            return False
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        
        # Low saturation + low contrast suggests haze/rain
        return mean_saturation < 60 and contrast < 40
    
    def _detect_shadows(self, image: np.ndarray) -> bool:
        """
        Detect strong shadow regions in the image.
        
        Uses the ratio of dark to bright pixels in the illumination channel.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute illumination via morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        illumination = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Shadow = regions where actual brightness << estimated illumination
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = gray.astype(float) / (illumination.astype(float) + 1e-6)
        
        shadow_pixels = np.sum(ratio < self.config.shadow_ratio_threshold)
        total_pixels = gray.size
        shadow_ratio = shadow_pixels / total_pixels
        
        return shadow_ratio > 0.15  # More than 15% of image in shadow
    
    # ──────────────────────────────────────────────
    # TARGETED ENHANCEMENT PIPELINES
    # ──────────────────────────────────────────────
    
    def _apply_enhancement(
        self, image: np.ndarray, degradation: DegradationType
    ) -> Tuple[np.ndarray, list]:
        """
        Apply targeted enhancement based on degradation type.
        
        Returns:
            (enhanced_image, list_of_enhancements_applied)
        """
        enhancements = []
        
        if degradation == DegradationType.CLEAR:
            return image.copy(), ["none (clear image)"]
        
        elif degradation == DegradationType.LOW_LIGHT:
            return self._enhance_low_light(image, enhancements)
        
        elif degradation == DegradationType.RAIN:
            return self._enhance_rain_haze(image, enhancements)
        
        elif degradation == DegradationType.SHADOW:
            return self._enhance_shadow(image, enhancements)
        
        elif degradation == DegradationType.BLUR:
            return self._enhance_blur(image, enhancements)
        
        elif degradation == DegradationType.GLARE:
            return self._enhance_glare(image, enhancements)
        
        return image.copy(), enhancements
    
    def _enhance_low_light(
        self, image: np.ndarray, enhancements: list
    ) -> Tuple[np.ndarray, list]:
        """
        Enhance low-light images using adaptive gamma + CLAHE.
        
        Pipeline:
            1. Adaptive gamma correction (brightens dark regions)
            2. CLAHE on L-channel (local contrast enhancement)
            3. Gentle denoising (low-light images are noisy)
        """
        result = image.copy()
        
        # Step 1: Adaptive gamma correction
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        # Compute gamma: darker images get lower gamma (more brightening)
        gamma = np.clip(
            np.log(127.5) / np.log(max(mean_brightness, 1.0)),
            self.config.gamma_low,
            self.config.gamma_high,
        )
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ]).astype("uint8")
        result = cv2.LUT(result, table)
        enhancements.append(f"adaptive_gamma({gamma:.2f})")
        
        # Step 2: CLAHE on L-channel (LAB color space)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhancements.append("clahe_lab")
        
        # Step 3: Gentle denoising
        result = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)
        enhancements.append("denoise_gentle")
        
        return result, enhancements
    
    def _enhance_rain_haze(
        self, image: np.ndarray, enhancements: list
    ) -> Tuple[np.ndarray, list]:
        """
        Enhance rain/haze images using guided filter dehazing + contrast boost.
        
        Pipeline:
            1. White balance correction
            2. Guided filter-based dehazing (Dark Channel Prior approximation)
            3. CLAHE contrast boost
            4. Saturation boost to recover colors
        """
        result = image.copy()
        
        # Step 1: Simple white balance (gray world assumption)
        result = self._white_balance(result)
        enhancements.append("white_balance")
        
        # Step 2: Simplified dehazing using dark channel prior
        result = self._dehaze(result)
        enhancements.append("dehaze_dcp")
        
        # Step 3: CLAHE contrast boost
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhancements.append("clahe_contrast")
        
        # Step 4: Saturation boost
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        enhancements.append("saturation_boost(1.3)")
        
        return result, enhancements
    
    def _enhance_shadow(
        self, image: np.ndarray, enhancements: list
    ) -> Tuple[np.ndarray, list]:
        """
        Enhance shadowy images using retinex-based illumination equalization.
        
        Pipeline:
            1. Multi-Scale Retinex (approximate) for shadow removal
            2. CLAHE on illumination-equalized result
        """
        result = image.copy()
        
        # Step 1: Single-scale Retinex approximation
        # R(x,y) = log(I(x,y)) - log(G(x,y) * I(x,y))
        result_float = result.astype(np.float32) + 1.0
        for sigma in [15, 80, 250]:
            blurred = cv2.GaussianBlur(result_float, (0, 0), sigma)
            retinex = np.log10(result_float) - np.log10(blurred + 1.0)
            # Normalize each channel
            for c in range(3):
                channel = retinex[:, :, c]
                channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6) * 255
                retinex[:, :, c] = channel
        
        result = np.clip(retinex, 0, 255).astype(np.uint8)
        enhancements.append("multi_scale_retinex")
        
        # Step 2: CLAHE
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhancements.append("clahe_shadow")
        
        return result, enhancements
    
    def _enhance_blur(
        self, image: np.ndarray, enhancements: list
    ) -> Tuple[np.ndarray, list]:
        """
        Enhance blurry images using unsharp masking + edge enhancement.
        
        Pipeline:
            1. Unsharp mask (Gaussian blur subtraction)
            2. Edge-aware sharpening using bilateral filter
        """
        result = image.copy()
        
        # Step 1: Unsharp mask
        gaussian = cv2.GaussianBlur(result, (0, 0), 3.0)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
        enhancements.append("unsharp_mask(1.5)")
        
        # Step 2: Edge-preserving detail enhancement
        try:
            result = cv2.detailEnhance(result, sigma_s=10, sigma_r=0.15)
            enhancements.append("detail_enhance")
        except Exception as e:
            logger.warning("cv2.detailEnhance failed (possibly OutOfMemory), falling back to unsharp mask only: %s", str(e))
        
        return result, enhancements
    
    def _enhance_glare(
        self, image: np.ndarray, enhancements: list
    ) -> Tuple[np.ndarray, list]:
        """
        Enhance images with glare/overexposure.
        
        Pipeline:
            1. Gamma correction (darken overexposed regions)
            2. Local contrast normalization
            3. CLAHE with lower clip limit
        """
        result = image.copy()
        
        # Step 1: Gamma correction to reduce brightness
        gamma = 1.8  # > 1 darkens
        table = np.array([
            ((i / 255.0) ** gamma) * 255 for i in range(256)
        ]).astype("uint8")
        result = cv2.LUT(result, table)
        enhancements.append(f"gamma_darken({gamma})")
        
        # Step 2: Local contrast normalization via CLAHE with lower clip
        clahe_low = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe_low.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhancements.append("clahe_glare(1.5)")
        
        return result, enhancements
    
    # ──────────────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────────────
    
    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply gray world white balance assumption."""
        result = image.astype(np.float32)
        avg_b, avg_g, avg_r = [np.mean(result[:, :, i]) for i in range(3)]
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / (avg_b + 1e-6)), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / (avg_g + 1e-6)), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / (avg_r + 1e-6)), 0, 255)
        
        return result.astype(np.uint8)
    
    def _dehaze(self, image: np.ndarray) -> np.ndarray:
        """
        Simplified dehazing using Dark Channel Prior.
        
        Reference: He et al., "Single Image Haze Removal Using Dark Channel Prior"
        """
        # Dark channel
        min_channel = np.min(image.astype(np.float32), axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(min_channel, kernel)
        
        # Estimate atmospheric light (top 0.1% brightest pixels in dark channel)
        flat_dark = dark_channel.flatten()
        num_pixels = max(int(flat_dark.size * 0.001), 1)
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        # Get corresponding pixels from original image
        rows, cols = np.unravel_index(indices, dark_channel.shape)
        atm_light = np.max(image[rows, cols], axis=0).astype(np.float32)
        atm_light = np.clip(atm_light, 1, 255)
        
        # Estimate transmission
        normalized = image.astype(np.float32) / atm_light
        min_normalized = np.min(normalized, axis=2)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        transmission = 1.0 - 0.95 * cv2.erode(min_normalized, kernel_small)
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # Recover scene
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = (
                (image[:, :, c].astype(np.float32) - atm_light[c]) / 
                transmission + atm_light[c]
            )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Final normalization: ensure valid range and consistent dtype."""
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _compute_quality_score(
        self, brightness: float, contrast: float, sharpness: float
    ) -> float:
        """
        Compute overall image quality score (0-1).
        
        Ideal ranges:
        - Brightness: 100-160
        - Contrast: 40-80
        - Sharpness: > 200
        """
        # Brightness score (bell curve centered at 130)
        b_score = np.exp(-((brightness - 130) ** 2) / (2 * 40 ** 2))
        
        # Contrast score (sigmoid, higher is better up to a point)
        c_score = min(contrast / 60.0, 1.0)
        
        # Sharpness score (sigmoid)
        s_score = min(sharpness / 300.0, 1.0)
        
        # Weighted combination
        quality = 0.3 * b_score + 0.3 * c_score + 0.4 * s_score
        return float(np.clip(quality, 0.0, 1.0))
