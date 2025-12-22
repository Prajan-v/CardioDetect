"""
Image Preprocessor for OCR Enhancement.
Addresses the limitation: "Tesseract OCR requires high-contrast headers"

Implements:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Adaptive Thresholding
- Noise Reduction
- Multi-pass OCR strategy
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Lazy imports to avoid startup overhead
cv2 = None
np = None
PIL_Image = None

def _ensure_cv2():
    """Lazy load OpenCV."""
    global cv2, np
    if cv2 is None:
        try:
            import cv2 as _cv2
            import numpy as _np
            cv2 = _cv2
            np = _np
        except ImportError:
            logger.warning("OpenCV not installed. Install with: pip install opencv-python")
            return False
    return True

def _ensure_pil():
    """Lazy load PIL."""
    global PIL_Image
    if PIL_Image is None:
        try:
            from PIL import Image
            PIL_Image = Image
        except ImportError:
            logger.warning("Pillow not installed. Install with: pip install Pillow")
            return False
    return True


class ImagePreprocessor:
    """
    Enhances document images for improved OCR accuracy.
    
    Strategies:
    1. CLAHE - Improves local contrast
    2. Adaptive Threshold - Binarizes for clean text
    3. Denoise - Removes noise artifacts
    4. Combined - All techniques together
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._cv2_available = _ensure_cv2()
        self._pil_available = _ensure_pil()
    
    def enhance_contrast_clahe(self, image) -> Optional[any]:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Best for: Low-contrast scanned documents, faded text.
        """
        if not self._cv2_available:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Create CLAHE object with tuned parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            if self.verbose:
                logger.info("CLAHE enhancement applied successfully")
            
            return enhanced
        except Exception as e:
            logger.error(f"CLAHE enhancement failed: {e}")
            return None
    
    def adaptive_threshold(self, image) -> Optional[any]:
        """
        Apply adaptive thresholding for text binarization.
        Best for: Mixed lighting conditions, gradients.
        """
        if not self._cv2_available:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold - better for varying lighting
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # Block size
                2    # C constant
            )
            
            if self.verbose:
                logger.info("Adaptive thresholding applied successfully")
            
            return binary
        except Exception as e:
            logger.error(f"Adaptive thresholding failed: {e}")
            return None
    
    def denoise(self, image) -> Optional[any]:
        """
        Apply noise reduction using bilateral filter.
        Best for: Noisy scans, JPEG artifacts.
        """
        if not self._cv2_available:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Bilateral filter preserves edges while reducing noise
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            if self.verbose:
                logger.info("Denoising applied successfully")
            
            return denoised
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return None
    
    def sharpen(self, image) -> Optional[any]:
        """
        Apply sharpening kernel to enhance text edges.
        Best for: Blurry scans, low-resolution images.
        """
        if not self._cv2_available:
            return None
        
        try:
            # Sharpening kernel
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            
            sharpened = cv2.filter2D(image, -1, kernel)
            
            if self.verbose:
                logger.info("Sharpening applied successfully")
            
            return sharpened
        except Exception as e:
            logger.error(f"Sharpening failed: {e}")
            return None
    
    def deskew(self, image) -> Optional[any]:
        """
        Correct document skew/rotation.
        Best for: Crooked scans, rotated documents.
        """
        if not self._cv2_available:
            return None
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Invert for contour detection
            inverted = cv2.bitwise_not(gray)
            
            # Find coordinates of non-zero pixels
            coords = np.column_stack(np.where(inverted > 0))
            
            if len(coords) > 100:
                # Get rotation angle
                angle = cv2.minAreaRect(coords)[-1]
                
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                
                # Rotate if significant skew detected
                if abs(angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(
                        image, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    if self.verbose:
                        logger.info(f"Deskewed by {angle:.2f} degrees")
                    return rotated
            
            return image
        except Exception as e:
            logger.error(f"Deskew failed: {e}")
            return None
    
    def preprocess_image(self, image_path: str) -> List[Tuple[str, any]]:
        """
        Apply multiple preprocessing strategies and return all variants.
        
        Returns:
            List of (strategy_name, processed_image) tuples
        """
        if not self._cv2_available:
            logger.warning("OpenCV not available, returning empty preprocessing results")
            return []
        
        variants = []
        
        try:
            # Load image
            original = cv2.imread(str(image_path))
            if original is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            variants.append(('original', original))
            
            # Strategy 1: CLAHE only
            clahe_result = self.enhance_contrast_clahe(original)
            if clahe_result is not None:
                variants.append(('clahe', clahe_result))
            
            # Strategy 2: Adaptive threshold only
            threshold_result = self.adaptive_threshold(original)
            if threshold_result is not None:
                variants.append(('adaptive_threshold', threshold_result))
            
            # Strategy 3: Denoise + CLAHE
            denoised = self.denoise(original)
            if denoised is not None:
                clahe_denoised = self.enhance_contrast_clahe(denoised)
                if clahe_denoised is not None:
                    variants.append(('denoise_clahe', clahe_denoised))
            
            # Strategy 4: Full pipeline (Denoise -> CLAHE -> Sharpen)
            if denoised is not None:
                enhanced = self.enhance_contrast_clahe(denoised)
                if enhanced is not None:
                    sharpened = self.sharpen(enhanced)
                    if sharpened is not None:
                        variants.append(('full_pipeline', sharpened))
            
            if self.verbose:
                logger.info(f"Generated {len(variants)} image variants for OCR")
            
            return variants
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return []
    
    def save_preprocessed_variants(self, image_path: str, output_dir: str = None) -> List[str]:
        """
        Preprocess image and save all variants to disk.
        
        Returns:
            List of paths to saved variant images
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='ocr_preprocessed_')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        variants = self.preprocess_image(image_path)
        saved_paths = []
        
        for name, image in variants:
            if image is not None:
                output_path = output_dir / f"{name}.png"
                cv2.imwrite(str(output_path), image)
                saved_paths.append(str(output_path))
        
        return saved_paths


def preprocess_document_for_ocr(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience function to preprocess a document for OCR.
    
    Returns:
        Dictionary with preprocessing results and temp file paths
    """
    preprocessor = ImagePreprocessor(verbose=verbose)
    
    result = {
        'original_path': file_path,
        'variants': [],
        'best_strategy': None,
        'success': False
    }
    
    try:
        saved_paths = preprocessor.save_preprocessed_variants(file_path)
        result['variants'] = saved_paths
        result['success'] = len(saved_paths) > 0
        
        # Default best strategy is 'full_pipeline' if available
        for path in saved_paths:
            if 'full_pipeline' in path:
                result['best_strategy'] = path
                break
        
        if result['best_strategy'] is None and saved_paths:
            result['best_strategy'] = saved_paths[-1]  # Use last (most processed)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# Global instance
image_preprocessor = ImagePreprocessor(verbose=False)
