"""
Multi-Engine OCR Module
=======================
Combines Tesseract and PaddleOCR for improved accuracy
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    PaddleOCR = None

# Try to import Tesseract
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None


class MultiEngineOCR:
    """
    Multi-engine OCR that combines results from Tesseract and PaddleOCR.
    Uses token-level confidence fusion for best results.
    """
    
    def __init__(self, use_gpu: bool = False, verbose: bool = False):
        self.verbose = verbose
        self.paddle_ocr = None
        self.has_paddle = HAS_PADDLEOCR
        self.has_tesseract = HAS_TESSERACT
        
        # Initialize PaddleOCR if available
        if HAS_PADDLEOCR:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en'
                )
                self.log("✓ PaddleOCR initialized")
            except Exception as e:
                self.log(f"✗ PaddleOCR init failed: {e}")
                self.has_paddle = False
        
        if HAS_TESSERACT:
            self.log("✓ Tesseract available")
        
        self.log(f"Engines: Tesseract={self.has_tesseract}, PaddleOCR={self.has_paddle}")
    
    def log(self, msg: str):
        if self.verbose:
            print(f"[MultiOCR] {msg}")
    
    def extract_tesseract(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text using Tesseract with detailed word info.
        """
        if not self.has_tesseract:
            return []
        
        try:
            # Get detailed data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            words = []
            for i, text in enumerate(data['text']):
                text = text.strip()
                if text:
                    conf = float(data['conf'][i])
                    if conf > 0:  # Valid confidence
                        words.append({
                            'text': text,
                            'confidence': conf / 100.0,
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i],
                            'source': 'tesseract'
                        })
            
            return words
        except Exception as e:
            self.log(f"Tesseract error: {e}")
            return []
    
    def extract_paddle(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text using PaddleOCR.
        """
        if not self.has_paddle or not self.paddle_ocr:
            return []
        
        try:
            result = self.paddle_ocr.ocr(img, cls=True)
            
            words = []
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        box = line[0]
                        text_info = line[1]
                        
                        text = text_info[0] if isinstance(text_info, (list, tuple)) else text_info
                        conf = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.9
                        
                        # Get bounding box
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        
                        # Split into words
                        for word in str(text).split():
                            word = word.strip()
                            if word:
                                words.append({
                                    'text': word,
                                    'confidence': float(conf),
                                    'x': min(x_coords),
                                    'y': min(y_coords),
                                    'w': max(x_coords) - min(x_coords),
                                    'h': max(y_coords) - min(y_coords),
                                    'source': 'paddle'
                                })
            
            return words
        except Exception as e:
            self.log(f"PaddleOCR error: {e}")
            return []
    
    def fuse_results(self, tesseract_words: List[Dict], paddle_words: List[Dict]) -> Tuple[str, float]:
        """
        Fuse results from multiple engines using confidence-weighted voting.
        
        Strategy:
        1. For each position, prefer the word with higher confidence
        2. If both engines agree, boost confidence
        3. Build final text from best words
        """
        if not tesseract_words and not paddle_words:
            return "", 0.0
        
        if not tesseract_words:
            text = " ".join(w['text'] for w in paddle_words)
            avg_conf = np.mean([w['confidence'] for w in paddle_words])
            return text, avg_conf
        
        if not paddle_words:
            text = " ".join(w['text'] for w in tesseract_words)
            avg_conf = np.mean([w['confidence'] for w in tesseract_words])
            return text, avg_conf
        
        # Sort both by y-position then x-position (reading order)
        tesseract_words = sorted(tesseract_words, key=lambda w: (w['y'], w['x']))
        paddle_words = sorted(paddle_words, key=lambda w: (w['y'], w['x']))
        
        # Simple fusion: take all words from Tesseract, supplement with Paddle
        all_words = []
        
        # Use Tesseract as base (usually better for structured text)
        for tw in tesseract_words:
            # Check if Paddle has a similar word with higher confidence
            best_match = None
            best_overlap = 0
            
            for pw in paddle_words:
                # Check spatial overlap
                overlap_x = max(0, min(tw['x'] + tw['w'], pw['x'] + pw['w']) - max(tw['x'], pw['x']))
                overlap_y = max(0, min(tw['y'] + tw['h'], pw['y'] + pw['h']) - max(tw['y'], pw['y']))
                overlap = overlap_x * overlap_y
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = pw
            
            if best_match and best_match['confidence'] > tw['confidence']:
                # Use Paddle's word
                word = best_match.copy()
                # Boost confidence if both agree
                if tw['text'].lower() == best_match['text'].lower():
                    word['confidence'] = min(1.0, (tw['confidence'] + best_match['confidence']) / 2 + 0.1)
                all_words.append(word)
            else:
                all_words.append(tw)
        
        # Build text
        text = " ".join(w['text'] for w in all_words)
        avg_conf = np.mean([w['confidence'] for w in all_words]) if all_words else 0.0
        
        return text, avg_conf
    
    def extract(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Extract text using all available engines and fuse results.
        """
        self.log("Starting multi-engine extraction...")
        
        results = {
            'tesseract': {'words': [], 'text': '', 'confidence': 0},
            'paddle': {'words': [], 'text': '', 'confidence': 0},
            'fused': {'text': '', 'confidence': 0},
            'engines_used': []
        }
        
        # Tesseract
        if self.has_tesseract:
            t_words = self.extract_tesseract(img)
            if t_words:
                results['tesseract']['words'] = t_words
                results['tesseract']['text'] = " ".join(w['text'] for w in t_words)
                results['tesseract']['confidence'] = np.mean([w['confidence'] for w in t_words])
                results['engines_used'].append('tesseract')
                self.log(f"Tesseract: {len(t_words)} words, {results['tesseract']['confidence']:.2%} conf")
        
        # PaddleOCR
        if self.has_paddle:
            p_words = self.extract_paddle(img)
            if p_words:
                results['paddle']['words'] = p_words
                results['paddle']['text'] = " ".join(w['text'] for w in p_words)
                results['paddle']['confidence'] = np.mean([w['confidence'] for w in p_words])
                results['engines_used'].append('paddle')
                self.log(f"PaddleOCR: {len(p_words)} words, {results['paddle']['confidence']:.2%} conf")
        
        # Fuse
        fused_text, fused_conf = self.fuse_results(
            results['tesseract']['words'],
            results['paddle']['words']
        )
        results['fused']['text'] = fused_text
        results['fused']['confidence'] = fused_conf
        
        self.log(f"Fused result: {fused_conf:.2%} confidence")
        
        return results
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract from image file.
        """
        img = cv2.imread(str(file_path))
        if img is None:
            return {'error': f"Could not read image: {file_path}"}
        
        return self.extract(img)


# Singleton
_engine = None

def get_multi_ocr(verbose: bool = False) -> MultiEngineOCR:
    global _engine
    if _engine is None:
        _engine = MultiEngineOCR(verbose=verbose)
    return _engine


if __name__ == '__main__':
    import sys
    
    engine = MultiEngineOCR(verbose=True)
    
    if len(sys.argv) > 1:
        result = engine.extract_from_file(sys.argv[1])
        print("\n=== Results ===")
        print(f"Engines used: {result['engines_used']}")
        print(f"\nTesseract confidence: {result['tesseract']['confidence']:.2%}")
        print(f"PaddleOCR confidence: {result['paddle']['confidence']:.2%}")
        print(f"Fused confidence: {result['fused']['confidence']:.2%}")
        print(f"\nFused text:\n{result['fused']['text'][:500]}...")
    else:
        print("Usage: python multi_ocr.py <image_path>")
        print(f"\nAvailable engines: Tesseract={engine.has_tesseract}, PaddleOCR={engine.has_paddle}")
