"""
Advanced OCR Enhancements
=========================
Additional OCR processing capabilities:
- Super-resolution upscaling
- Adaptive thresholding
- Table detection and extraction
- Layout analysis
- Handwriting recognition support
- Language model post-processing
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import re

# Try to import optional dependencies
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class SuperResolution:
    """
    Super-resolution upscaling for low-quality images.
    Uses classical interpolation + sharpening (no deep learning for speed).
    """
    
    @staticmethod
    def upscale(image: np.ndarray, scale: float = 2.0, method: str = 'lanczos') -> np.ndarray:
        """
        Upscale image with high-quality interpolation.
        
        Args:
            image: Input image
            scale: Scale factor (2.0 = double size)
            method: 'lanczos', 'cubic', or 'linear'
        """
        if scale <= 1.0:
            return image
        
        h, w = image.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        
        interpolation = {
            'lanczos': cv2.INTER_LANCZOS4,
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR
        }.get(method, cv2.INTER_LANCZOS4)
        
        upscaled = cv2.resize(image, new_size, interpolation=interpolation)
        
        # Apply sharpening after upscale
        kernel = np.array([[-0.5, -1, -0.5],
                          [-1,   7,  -1],
                          [-0.5, -1, -0.5]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        return sharpened
    
    @staticmethod
    def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Full enhancement pipeline for OCR.
        """
        # Auto-detect if upscaling needed (low DPI check)
        h, w = image.shape[:2]
        
        # If image is small, upscale
        if w < 1000 or h < 800:
            image = SuperResolution.upscale(image, scale=2.0)
        
        # Apply CLAHE for local contrast enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        
        return image


class AdaptiveThreshold:
    """
    Intelligent adaptive thresholding that auto-selects the best method.
    """
    
    @staticmethod
    def analyze_image(gray: np.ndarray) -> Dict[str, float]:
        """Analyze image characteristics to choose best thresholding."""
        # Calculate statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Calculate local contrast
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_contrast = np.mean(np.abs(gray.astype(np.float32) - local_mean))
        
        # Calculate noise level (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        return {
            'mean': mean_val,
            'std': std_val,
            'local_contrast': local_contrast,
            'noise': noise_level
        }
    
    @staticmethod
    def auto_threshold(gray: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Automatically select and apply best thresholding method.
        
        Returns:
            Tuple of (thresholded image, method used)
        """
        stats = AdaptiveThreshold.analyze_image(gray)
        
        # Decision tree for method selection
        if stats['noise'] > 500:
            # High noise - use Gaussian adaptive
            method = 'gaussian_adaptive'
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 15
            )
        elif stats['local_contrast'] < 20:
            # Low contrast - use CLAHE + Otsu
            method = 'clahe_otsu'
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif stats['std'] < 30:
            # Uniform lighting - use mean adaptive
            method = 'mean_adaptive'
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 15, 10
            )
        else:
            # Standard case - use Otsu
            method = 'otsu'
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary, method
    
    @staticmethod
    def multi_threshold(gray: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Apply multiple thresholding methods and return all results.
        Useful for ensemble OCR.
        """
        results = []
        
        # Otsu's
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append((otsu, 'otsu'))
        
        # Gaussian Adaptive
        gauss = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 11
        )
        results.append((gauss, 'gaussian'))
        
        # Mean Adaptive
        mean_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, 10
        )
        results.append((mean_thresh, 'mean'))
        
        # Sauvola-like (local)
        kernel_size = 25
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        std = np.sqrt(cv2.blur(gray.astype(np.float32)**2, (kernel_size, kernel_size)) - mean.astype(np.float32)**2)
        k = 0.2
        threshold = mean * (1 + k * (std / 128 - 1))
        sauvola = ((gray > threshold) * 255).astype(np.uint8)
        results.append((sauvola, 'sauvola'))
        
        return results


class TableExtractor:
    """
    Extract tables from document images.
    Uses line detection and cell segmentation.
    """
    
    @staticmethod
    def detect_lines(gray: np.ndarray, min_length: int = 50) -> Tuple[List, List]:
        """Detect horizontal and vertical lines."""
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        
        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        # Find contours
        h_contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return h_contours, v_contours
    
    @staticmethod
    def find_table_regions(image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find table regions in image.
        
        Returns:
            List of table regions with bounding boxes and structure
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        h_contours, v_contours = TableExtractor.detect_lines(gray)
        
        # If not enough lines for a table, return empty
        if len(h_contours) < 2 or len(v_contours) < 2:
            return []
        
        # Combine lines to find table boundaries
        h_lines = cv2.drawContours(np.zeros_like(gray), h_contours, -1, 255, 2)
        v_lines = cv2.drawContours(np.zeros_like(gray), v_contours, -1, 255, 2)
        
        # Find intersection points (table cells)
        combined = cv2.bitwise_or(h_lines, v_lines)
        
        # Find table bounding box
        points = np.column_stack(np.where(combined > 0))
        if len(points) < 4:
            return []
        
        x, y, tw, th = cv2.boundingRect(points)
        
        tables = [{
            'bbox': {'x': y, 'y': x, 'width': th, 'height': tw},
            'rows': len(h_contours) - 1,
            'cols': len(v_contours) - 1,
            'h_lines': len(h_contours),
            'v_lines': len(v_contours)
        }]
        
        return tables
    
    @staticmethod
    def extract_cells(image: np.ndarray, table: Dict) -> List[np.ndarray]:
        """Extract individual cell images from a table region."""
        bbox = table['bbox']
        roi = image[bbox['y']:bbox['y']+bbox['height'], 
                   bbox['x']:bbox['x']+bbox['width']]
        
        # For now, return the whole table region
        # Full implementation would segment into cells
        return [roi]


class LayoutAnalyzer:
    """
    Analyze document layout to identify regions.
    Simplified version (no deep learning model).
    """
    
    @staticmethod
    def detect_text_regions(image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions using connected components.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter small regions
            if area > 1000:
                aspect_ratio = w / h if h > 0 else 0
                
                # Classify region type
                if aspect_ratio > 5:
                    region_type = 'text_line'
                elif aspect_ratio > 2:
                    region_type = 'text_block'
                elif 0.8 < aspect_ratio < 1.2:
                    region_type = 'figure'
                else:
                    region_type = 'unknown'
                
                regions.append({
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'type': region_type,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
        
        # Sort by position (top to bottom, left to right)
        regions.sort(key=lambda r: (r['bbox']['y'], r['bbox']['x']))
        
        return regions
    
    @staticmethod
    def detect_key_value_pairs(image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect key-value pairs (e.g., "Age: 55").
        """
        regions = LayoutAnalyzer.detect_text_regions(image)
        
        # Filter for horizontal text blocks (likely key-value)
        kv_candidates = [r for r in regions if r['type'] in ['text_line', 'text_block']]
        
        return kv_candidates


class HandwritingSupport:
    """
    Support for handwriting detection and processing.
    Uses preprocessing to improve Tesseract on handwritten text.
    """
    
    @staticmethod
    def is_handwritten(image: np.ndarray) -> bool:
        """
        Heuristic to detect if image contains handwriting.
        Handwritten text typically has more variance in stroke width.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate stroke width variance using distance transform
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        if np.sum(binary > 0) > 0:
            stroke_variance = np.std(dist[binary > 0])
        else:
            stroke_variance = 0
        
        # Heuristic: handwriting has higher stroke variance
        return stroke_variance > 2.5 and edge_density > 0.05
    
    @staticmethod
    def preprocess_handwriting(image: np.ndarray) -> np.ndarray:
        """
        Preprocess handwritten text for better OCR.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 2. Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Binarize with Sauvola-like method (better for handwriting)
        kernel_size = 31
        mean = cv2.blur(enhanced.astype(np.float32), (kernel_size, kernel_size))
        std = np.sqrt(cv2.blur(enhanced.astype(np.float32)**2, (kernel_size, kernel_size)) - mean**2)
        k = 0.3
        threshold = mean * (1 + k * (std / 128 - 1))
        binary = ((enhanced > threshold) * 255).astype(np.uint8)
        
        # 4. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned


class LanguageModelCorrector:
    """
    Post-OCR correction using language model patterns.
    Uses regex and heuristics (no deep learning for speed).
    """
    
    def __init__(self):
        # Common OCR error patterns
        self.error_patterns = [
            (r'\bl\b', '1'),  # Single 'l' often should be '1' in values
            (r'(\d)l(\d)', r'\g<1>1\g<2>'),  # 'l' between digits -> '1'
            (r'(\d)O(\d)', r'\g<1>0\g<2>'),  # 'O' between digits -> '0'
            (r'(\d)o(\d)', r'\g<1>0\g<2>'),  # 'o' between digits -> '0'
            (r'(\d)S(\d)', r'\g<1>5\g<2>'),  # 'S' between digits -> '5'
            (r'(\d)s(\d)', r'\g<1>5\g<2>'),  # 's' between digits -> '5'
            (r'rng/dl', 'mg/dL'),  # Common unit error
            (r'mg/d1', 'mg/dL'),
            (r'mm Hg', 'mmHg'),
            (r'mm hg', 'mmHg'),
            (r'MMHG', 'mmHg'),
        ]
        
        # Medical term corrections
        self.term_corrections = {
            'cholestero1': 'cholesterol',
            'g1ucose': 'glucose',
            'diabe tes': 'diabetes',
            'hyper tension': 'hypertension',
            'b1ood': 'blood',
            'cardio vascular': 'cardiovascular',
        }
    
    def correct(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Apply language model corrections to OCR text.
        
        Returns:
            Tuple of (corrected text, list of corrections made)
        """
        corrections = []
        corrected = text
        
        # Apply regex patterns
        for pattern, replacement in self.error_patterns:
            matches = re.findall(pattern, corrected)
            if matches:
                corrected = re.sub(pattern, replacement, corrected)
                corrections.append({
                    'type': 'pattern',
                    'pattern': pattern,
                    'replacement': replacement,
                    'count': len(matches)
                })
        
        # Apply term corrections
        for wrong, right in self.term_corrections.items():
            if wrong.lower() in corrected.lower():
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                corrected = pattern.sub(right, corrected)
                corrections.append({
                    'type': 'term',
                    'original': wrong,
                    'corrected': right
                })
        
        # Fix common number-unit spacing
        corrected = re.sub(r'(\d)\s+(mg|mmHg|%|bpm)', r'\1\2', corrected)
        
        return corrected, corrections
    
    def validate_medical_values(self, text: str) -> List[Dict]:
        """
        Validate extracted medical values are in reasonable ranges.
        """
        warnings = []
        
        # Extract and validate values
        patterns = [
            (r'age[:\s]+(\d+)', 'age', 1, 120),
            (r'systolic[:\s]+(\d+)', 'systolic', 60, 250),
            (r'diastolic[:\s]+(\d+)', 'diastolic', 40, 150),
            (r'cholesterol[:\s]+(\d+)', 'cholesterol', 50, 500),
            (r'glucose[:\s]+(\d+)', 'glucose', 30, 500),
            (r'hdl[:\s]+(\d+)', 'hdl', 10, 150),
            (r'ldl[:\s]+(\d+)', 'ldl', 20, 300),
        ]
        
        for pattern, field, min_val, max_val in patterns:
            match = re.search(pattern, text.lower())
            if match:
                value = int(match.group(1))
                if value < min_val or value > max_val:
                    warnings.append({
                        'field': field,
                        'value': value,
                        'expected_range': f'{min_val}-{max_val}',
                        'issue': 'out_of_range'
                    })
        
        return warnings


# Convenience functions
def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Full enhancement pipeline."""
    return SuperResolution.enhance_for_ocr(image)

def auto_threshold(gray: np.ndarray) -> Tuple[np.ndarray, str]:
    """Automatic thresholding."""
    return AdaptiveThreshold.auto_threshold(gray)

def detect_tables(image: np.ndarray) -> List[Dict]:
    """Detect tables in image."""
    return TableExtractor.find_table_regions(image)

def analyze_layout(image: np.ndarray) -> List[Dict]:
    """Analyze document layout."""
    return LayoutAnalyzer.detect_text_regions(image)

def preprocess_handwriting(image: np.ndarray) -> np.ndarray:
    """Preprocess handwritten text."""
    return HandwritingSupport.preprocess_handwriting(image)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read: {image_path}")
            sys.exit(1)
        
        print(f"Image size: {image.shape}")
        
        # Test super resolution
        enhanced = SuperResolution.enhance_for_ocr(image)
        print(f"Enhanced size: {enhanced.shape}")
        
        # Test adaptive threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary, method = AdaptiveThreshold.auto_threshold(gray)
        print(f"Thresholding method selected: {method}")
        
        # Test table detection
        tables = TableExtractor.find_table_regions(image)
        print(f"Tables detected: {len(tables)}")
        
        # Test layout analysis
        regions = LayoutAnalyzer.detect_text_regions(image)
        print(f"Text regions detected: {len(regions)}")
        
        # Test handwriting detection
        is_hw = HandwritingSupport.is_handwritten(image)
        print(f"Handwriting detected: {is_hw}")
        
        # Test language model
        lm = LanguageModelCorrector()
        test_text = "Age: 55 years, Cho1estero1: 245 rng/dl, B1ood Pressure: 140/90 mm Hg"
        corrected, corrections = lm.correct(test_text)
        print(f"\nOriginal: {test_text}")
        print(f"Corrected: {corrected}")
        print(f"Corrections: {len(corrections)}")
    else:
        print("Usage: python advanced_ocr.py <image_path>")
        print("\nModules available:")
        print("  - SuperResolution: Upscale low-DPI images")
        print("  - AdaptiveThreshold: Auto-select best thresholding")
        print("  - TableExtractor: Detect and extract tables")
        print("  - LayoutAnalyzer: Detect text regions")
        print("  - HandwritingSupport: Preprocess handwritten text")
        print("  - LanguageModelCorrector: Post-OCR correction")
