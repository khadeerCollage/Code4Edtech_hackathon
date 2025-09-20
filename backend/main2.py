import os
import cv2
import numpy as np
import json
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

try:
    import pytesseract
    # Allow environment variable override for Windows
    _tess_cmd = os.environ.get('TESSERACT_CMD')
    if _tess_cmd and os.path.exists(_tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
except Exception:
    pytesseract = None

# New utility to check OCR availability
def _ocr_available() -> bool:
    return pytesseract is not None and hasattr(pytesseract, 'image_to_string')

# ==================== CONFIGURATION ====================

class OMRConfig:
    """Central configuration for OMR sheet processing"""
    
    # Sheet Layout
    TOTAL_QUESTIONS = 100
    CHOICES_PER_QUESTION = 4  # A, B, C, D
    QUESTIONS_PER_COLUMN = 20
    COLUMNS_PER_SHEET = 5
    
    # Bubble Detection Parameters
    BUBBLE_RADIUS = 12
    BUBBLE_MIN_AREA = 30   # Reduced from 50 for smaller bubbles
    BUBBLE_MAX_AREA = 800  # Increased from 500 for larger bubbles
    
    # Adaptive Threshold for different marking styles
    MARKING_THRESHOLD = 0.12  # Reduced from 0.15 to be more sensitive
    
    # Subject Detection Keywords
    SUBJECT_KEYWORDS = [
        "PYTHON", "DATA ANALYSIS", "MYSQL", "POWER BI", "ADV STATS",
        "STATISTICS", "MACHINE LEARNING", "DEEP LEARNING", 
        "DATA SCIENCE", "SQL", "TABLEAU", "EXCEL"
    ]
    
    # OCR Configuration
    OCR_CONFIG = '--psm 6 --oem 3'  # Best for uniform text blocks
    
    # Debug Mode
    DEBUG = True  # Enable debug visualization

# ==================== DATASET MANAGEMENT ====================

class DatasetManager:
    """Handles dataset discovery and path management"""
    
    def __init__(self):
        self.root = self._find_dataset_root()
        self.output_dir = self.root / 'results'
        self.key_file = self._find_key_file()
        self.sets = self._discover_sets()
        
    def _find_dataset_root(self) -> Path:
        """Find dataset root directory"""
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / 'dataset' / 'data',
            here.parent / 'dataset' / 'Theme 1 - Sample Data',
            here.parent / 'dataset'
        ]
        
        for cand in candidates:
            if cand.exists():
                for p in cand.iterdir():
                    if p.is_dir() and 'set' in p.name.lower():
                        return cand
        return candidates[0]
    
    def _find_key_file(self) -> Optional[Path]:
        """Find answer key Excel file"""
        patterns = ['*key*.xlsx', '*Key*.xlsx', '*KEY*.xlsx']
        for pattern in patterns:
            files = list(self.root.glob(pattern))
            if files:
                return files[0]
        return None
    
    def _discover_sets(self) -> Dict[str, Path]:
        """Discover exam set folders"""
        sets = {}
        for p in self.root.iterdir():
            if p.is_dir():
                name_lower = p.name.lower().replace('-', ' ').replace('_', ' ')
                if 'set a' in name_lower or 'a' == name_lower.strip():
                    sets['Set_A'] = p
                elif 'set b' in name_lower or 'b' == name_lower.strip():
                    sets['Set_B'] = p
        return sets

# ==================== IMAGE PREPROCESSING ====================

class ImagePreprocessor:
    """Handles image preprocessing and perspective correction"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return denoised
    
    @staticmethod
    def find_sheet_contour(image: np.ndarray) -> Optional[np.ndarray]:
        """Find the OMR sheet contour"""
        gray = ImagePreprocessor.enhance_image(image)
        
        # Edge detection with multiple thresholds
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 30, 100)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find rectangular contour
        for cnt in contours[:5]:  # Check top 5 largest contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order points in consistent sequence: TL, TR, BR, BL"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and diff for corner detection
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect
    
    @staticmethod
    def transform_perspective(image: np.ndarray, width: int = 800, height: int = 1200) -> Optional[np.ndarray]:
        """Apply perspective transformation to get top-down view"""
        sheet_contour = ImagePreprocessor.find_sheet_contour(image)
        if sheet_contour is None:
            # Fallback: resize original to target size for downstream rough processing
            try:
                fallback = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                # Attach attribute to signal fallback usage (cannot on np.ndarray directly; return tuple instead)
                return fallback  # Caller will infer by separate flag
            except Exception:
                return None
        
        # Order points
        src_pts = ImagePreprocessor.order_points(sheet_contour)
        
        # Destination points
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Get transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transformation
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped

# ==================== OCR ENGINE ====================

class OCREngine:
    """Advanced OCR for text and structure detection"""
    
    def __init__(self):
        self.config = OMRConfig()
        
    def extract_set_code(self, image: np.ndarray) -> Optional[str]:
        """Extract exam set code with enhanced pattern matching and visual analysis"""
        if not _ocr_available():
            return None
        
        try:
            h, w = image.shape[:2]
            # Focus on top-left area where set info appears
            header = image[0:int(h*0.15), 0:int(w*0.7)]  # Top 15%, left 70%
            
            # Multiple preprocessing approaches
            gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY) if len(header.shape) == 3 else header
            
            # Strategy 1: High contrast for handwritten text
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
            enhanced = clahe.apply(gray)
            
            # Strategy 2: Threshold variations
            preprocessed_images = []
            
            # OTSU
            _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(binary1)
            
            # Inverted OTSU (for dark text on light)
            _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            preprocessed_images.append(binary2)
            
            # Enhanced version
            _, binary3 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
            preprocessed_images.append(binary3)
            
            # Adaptive threshold
            binary4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            preprocessed_images.append(binary4)
            
            all_candidates = []
            
            # Try different OCR configurations on each preprocessed image
            for i, img in enumerate(preprocessed_images):
                configs = [
                    '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-_',
                    '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-_',
                    '--psm 6 --oem 3',
                    '--psm 13 --oem 3'  # Raw line for handwriting
                ]
                
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(img, config=config).strip()
                        if text:
                            all_candidates.append(text.upper())
                    except:
                        continue
            
            # Also try to get structured data to find specific regions
            try:
                data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                for i, txt in enumerate(data['text']):
                    if txt and txt.strip():
                        all_candidates.append(txt.strip().upper())
            except:
                pass
            
            # Process all candidates with extensive pattern matching
            for text in set(all_candidates):  # Remove duplicates
                text_clean = text.replace(' ', '').replace('\n', '').replace('\t', '')
                
                # Direct SET-A/SET-B patterns
                patterns = [
                    r'SET[:\-_\s]*([AB])\b',
                    r'SETNO[:\-_\s]*([AB])\b',
                    r'SET[:\-_\s]*([AB])$',
                    r'\bSET([AB])\b',
                    r'SET[:\-_]*([AB])',
                    r'([AB])\s*$',  # Just A or B at end
                    r'^([AB])\b',   # A or B at start
                    r'SET.*?([AB])',
                    r'S[E3]T[:\-_\s]*([AB])',  # OCR might read E as 3
                    r'5[E3]T[:\-_\s]*([AB])',  # OCR might read S as 5
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text_clean)
                    if match and match.group(1) in ['A', 'B']:
                        return match.group(1)
                
                # Direct string matching for common OCR errors
                error_mappings = {
                    'SETA': 'A', 'SET_A': 'A', 'SET-A': 'A', 'SET:A': 'A',
                    'SETB': 'B', 'SET_B': 'B', 'SET-B': 'B', 'SET:B': 'B',
                    'S3TA': 'A', '5ETA': 'A', 'SFTA': 'A',  # Common OCR errors
                    'S3TB': 'B', '5ETB': 'B', 'SFTB': 'B'
                }
                
                for pattern, result in error_mappings.items():
                    if pattern in text_clean:
                        return result
                
                # Look for A or B near "SET" keyword
                if 'SET' in text_clean:
                    # Extract characters around SET
                    set_pos = text_clean.find('SET')
                    context = text_clean[max(0, set_pos-3):set_pos+10]
                    for char in ['A', 'B']:
                        if char in context:
                            return char
                
                # Last resort: if we see clear A or B in the text
                if text_clean == 'A' or text_clean == 'B':
                    return text_clean
                
                # Check for A or B within first few characters (might be isolated)
                first_chars = text_clean[:5]
                for char in ['A', 'B']:
                    if char in first_chars:
                        return char
            
            # Visual analysis fallback - look for handwritten A or B patterns
            # This would be where we could add shape-based recognition
            
        except Exception as e:
            print(f"Enhanced set code extraction error: {e}")
        
        return None
    
    def detect_subjects(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """Detect subject blocks and their positions"""
        if not _ocr_available():
            return {}
        
        try:
            # Get OCR data with bounding boxes
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            subjects = {}
            found_positions = []
            
            # Find subject keywords
            for i, text in enumerate(ocr_data['text']):
                if not text:
                    continue
                    
                text_upper = text.upper().strip()
                for keyword in self.config.SUBJECT_KEYWORDS:
                    if keyword in text_upper or text_upper in keyword:
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        found_positions.append({
                            'name': keyword,
                            'y': y,
                            'x': x,
                            'w': w,
                            'h': h
                        })
                        break
            
            # Sort by vertical position
            found_positions.sort(key=lambda p: p['y'])
            
            # Define subject blocks
            img_h, img_w = image.shape[:2]
            for i, pos in enumerate(found_positions):
                y_start = pos['y']
                y_end = found_positions[i+1]['y'] if i+1 < len(found_positions) else img_h
                subjects[pos['name']] = (0, y_start, img_w, y_end)
            
            return subjects
            
        except Exception as e:
            print(f"Subject detection error: {e}")
            return {}
    
    def find_question_structure(self, image: np.ndarray, subject_blocks: Dict) -> Dict:
        """Find question numbers and option positions"""
        if not _ocr_available() or not subject_blocks:
            return {}
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            structure = {}
            
            for subject, (x1, y1, x2, y2) in subject_blocks.items():
                # Find question numbers in this block
                questions = []
                options = {'A': [], 'B': [], 'C': [], 'D': []}
                
                for i, text in enumerate(ocr_data['text']):
                    if not text:
                        continue
                    
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    
                    # Check if in current subject block
                    if not (x1 <= x <= x2 and y1 <= y <= y2):
                        continue
                    
                    # Check for question number (1-100)
                    if text.isdigit():
                        q_num = int(text)
                        if 1 <= q_num <= 100:
                            questions.append({
                                'num': q_num,
                                'x': x,
                                'y': y,
                                'w': ocr_data['width'][i],
                                'h': ocr_data['height'][i]
                            })
                    
                    # Check for option letters
                    elif text.upper() in ['A', 'B', 'C', 'D']:
                        options[text.upper()].append({
                            'x': x,
                            'y': y,
                            'w': ocr_data['width'][i],
                            'h': ocr_data['height'][i]
                        })
                
                # Match questions with options
                for q in questions:
                    q_structure = {'question': q['num'], 'options': {}}
                    
                    # Find closest option headers to this question
                    for opt_letter, opt_positions in options.items():
                        if opt_positions:
                            # Find option at similar Y level
                            closest = min(opt_positions, 
                                        key=lambda p: abs(p['y'] - q['y']))
                            if abs(closest['y'] - q['y']) < 50:  # Within 50 pixels
                                q_structure['options'][opt_letter] = {
                                    'x': closest['x'],
                                    'y': closest['y'],
                                    'w': closest['w'],
                                    'h': closest['h']
                                }
                    
                    if q_structure['options']:
                        structure[str(q['num'])] = q_structure
            
            return structure
            
        except Exception as e:
            print(f"Structure detection error: {e}")
            return {}

# ==================== BUBBLE DETECTOR ====================

class BubbleDetector:
    """Advanced bubble detection with multiple strategies"""
    
    def __init__(self):
        self.config = OMRConfig()
    
    def create_bubble_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask for bubble detection using multiple techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Multiple threshold strategies
        masks = []
        
        # Adaptive threshold (good for varying lighting)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 10)
        masks.append(adaptive)
        
        # OTSU threshold (good for uniform lighting)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(otsu)
        
        # Combine masks
        combined = np.zeros_like(gray)
        for mask in masks:
            combined = cv2.bitwise_or(combined, mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def detect_marked_bubble(self, image: np.ndarray, option_positions: Dict) -> Optional[str]:
        """Detect which bubble is marked for a question"""
        if not option_positions:
            return None
        
        mask = self.create_bubble_mask(image)
        
        bubble_scores = {}
        
        for option, pos in option_positions.items():
            # Bubble is BELOW the option text
            bubble_x = pos['x'] + pos['w'] // 2
            bubble_y = pos['y'] + pos['h'] + 25  # Adjust offset as needed
            
            # Create ROI around expected bubble position
            roi_size = self.config.BUBBLE_RADIUS * 2
            x1 = max(0, bubble_x - roi_size)
            y1 = max(0, bubble_y - roi_size)
            x2 = min(mask.shape[1], bubble_x + roi_size)
            y2 = min(mask.shape[0], bubble_y + roi_size)
            
            roi = mask[y1:y2, x1:x2]
            
            # Calculate fill ratio
            total_pixels = roi.size
            filled_pixels = cv2.countNonZero(roi)
            fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
            
            bubble_scores[option] = fill_ratio
        
        # Find marked bubble (highest fill ratio above threshold)
        max_option = max(bubble_scores.items(), key=lambda x: x[1])
        if max_option[1] >= self.config.MARKING_THRESHOLD:
            return max_option[0]
        
        return None
    
    def detect_bubbles_ocr_based(self, image: np.ndarray) -> Dict[str, str]:
        """Pure OCR-based bubble detection using human-like analysis"""
        if not _ocr_available():
            return self.detect_bubbles_grid(image)
        
        try:
            # Step 1: Enhanced image preprocessing for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Use the best preprocessing found: binary threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Use the best OCR config: Sparse text detection
            config = '--psm 11 --oem 3'  # Sparse text
            ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config=config)
            
            # Alternative config for option letters specifically 
            config_letters = '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCD'
            ocr_letters = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config=config_letters)
            
            # Step 2: Find all question numbers and option letters
            questions = {}  # {q_num: {'x': x, 'y': y, 'options': {}}}
            option_letters = []  # [{'letter': 'A', 'x': x, 'y': y, 'w': w, 'h': h}]
            
            # Process main OCR data for question numbers
            for i, text in enumerate(ocr_data['text']):
                if not text or not text.strip():
                    continue
                    
                text_clean = text.strip()
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Detect question numbers (1-100)
                if text_clean.isdigit():
                    q_num = int(text_clean)
                    if 1 <= q_num <= 100:
                        questions[q_num] = {
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'options': {}
                        }
                
                # Detect option letters from main OCR too
                elif len(text_clean) == 1 and text_clean.upper() in ['A', 'B', 'C', 'D']:
                    option_letters.append({
                        'letter': text_clean.upper(),
                        'x': x, 'y': y, 'w': w, 'h': h
                    })
            
            # Process letter-specific OCR data
            for i, text in enumerate(ocr_letters['text']):
                if not text or not text.strip():
                    continue
                    
                text_clean = text.strip()
                x = ocr_letters['left'][i]
                y = ocr_letters['top'][i]
                w = ocr_letters['width'][i]
                h = ocr_letters['height'][i]
                
                if len(text_clean) == 1 and text_clean.upper() in ['A', 'B', 'C', 'D']:
                    option_letters.append({
                        'letter': text_clean.upper(),
                        'x': x, 'y': y, 'w': w, 'h': h
                    })
            
            print(f"OCR Detection: {len(questions)} questions, {len(option_letters)} option letters")
            
            # If we have very few detections, try region-based OCR
            if len(questions) < 20 or len(option_letters) < 40:
                print("Low detection count, trying region-based OCR...")
                regions_questions, regions_options = self.ocr_by_regions(image)
                
                # Merge results
                questions.update(regions_questions)
                option_letters.extend(regions_options)
                
                print(f"After region OCR: {len(questions)} questions, {len(option_letters)} option letters")
            
            # Step 3: Associate option letters with questions based on spatial proximity
            for q_num, q_data in questions.items():
                q_x, q_y = q_data['x'], q_data['y']
                
                # Find option letters near this question (within reasonable distance)
                nearby_options = []
                for opt in option_letters:
                    # Options should be to the right of question number and roughly same Y level
                    if (opt['x'] > q_x and 
                        abs(opt['y'] - q_y) < 60 and  # More lenient vertical distance
                        opt['x'] - q_x < 800):  # More lenient horizontal distance
                        nearby_options.append(opt)
                
                # Sort options by X coordinate (A, B, C, D should be left to right)
                nearby_options.sort(key=lambda o: o['x'])
                
                # Assign to question
                for opt in nearby_options:
                    questions[q_num]['options'][opt['letter']] = {
                        'x': opt['x'], 'y': opt['y'], 
                        'w': opt['w'], 'h': opt['h']
                    }
            
            # Step 4: Detect filled bubbles using contour analysis
            filled_bubbles = self.find_filled_bubbles(image)
            
            print(f"Found {len(filled_bubbles)} filled bubbles")
            
            # Step 5: Map filled bubbles to questions and options
            answers = {}
            
            for q_num in range(1, 101):  # Check all 100 questions
                if q_num not in questions:
                    answers[str(q_num)] = "NONE"
                    continue
                
                q_data = questions[q_num]
                if not q_data['options']:
                    # If no options detected, fall back to positional estimation
                    answers[str(q_num)] = self.estimate_answer_positionally(q_num, q_data, filled_bubbles, image)
                    continue
                
                # Find filled bubbles near each option
                marked_options = []
                
                for opt_letter, opt_data in q_data['options'].items():
                    opt_x = opt_data['x'] + opt_data['w'] // 2  # Center of option letter
                    opt_y = opt_data['y'] + opt_data['h']  # Below the option letter
                    
                    # Look for filled bubbles near this option
                    best_bubble = None
                    min_distance = float('inf')
                    
                    for bubble in filled_bubbles:
                        # Calculate distance from option position to bubble
                        dx = bubble['x'] - opt_x
                        dy = bubble['y'] - (opt_y + 25)  # Expect bubble ~25px below letter
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # More lenient distance matching
                        if distance < 50 and distance < min_distance:  # Increased threshold
                            min_distance = distance
                            best_bubble = bubble
                    
                    if best_bubble:
                        marked_options.append(opt_letter)
                
                # Determine final answer based on marked options
                if len(marked_options) == 0:
                    answers[str(q_num)] = "NONE"
                elif len(marked_options) == 1:
                    answers[str(q_num)] = marked_options[0]
                else:
                    # Multiple marks - human logic says this is wrong
                    answers[str(q_num)] = "MULTIPLE"
            
            return answers
            
        except Exception as e:
            print(f"OCR-based detection failed: {e}")
            return self.detect_bubbles_hybrid(image)

    def ocr_by_regions(self, image):
        """Try OCR on specific regions of the sheet"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        h, w = gray.shape
        questions = {}
        option_letters = []
        
        # Define regions for each column of questions
        cols = [
            (0.05, 0.25, 0.1, 0.9),   # Column 1 (Q1-Q20)
            (0.20, 0.40, 0.1, 0.9),   # Column 2 (Q21-Q40) 
            (0.35, 0.55, 0.1, 0.9),   # Column 3 (Q41-Q60)
            (0.50, 0.70, 0.1, 0.9),   # Column 4 (Q61-Q80)
            (0.65, 0.85, 0.1, 0.9),   # Column 5 (Q81-Q100)
        ]
        
        for col_idx, (x1, x2, y1, y2) in enumerate(cols):
            region = thresh[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            
            # OCR for questions
            try:
                config = '--psm 11 --oem 3'
                region_data = pytesseract.image_to_data(region, output_type=pytesseract.Output.DICT, config=config)
                
                for i, text in enumerate(region_data['text']):
                    if not text or not text.strip():
                        continue
                        
                    text_clean = text.strip()
                    x = region_data['left'][i] + int(w*x1)  # Adjust to full image coordinates
                    y = region_data['top'][i] + int(h*y1)
                    width = region_data['width'][i]
                    height = region_data['height'][i]
                    
                    # Question numbers
                    if text_clean.isdigit():
                        q_num = int(text_clean)
                        if 1 <= q_num <= 100:
                            questions[q_num] = {
                                'x': x, 'y': y, 'w': width, 'h': height,
                                'options': {}
                            }
                    
                    # Option letters
                    elif len(text_clean) == 1 and text_clean.upper() in ['A', 'B', 'C', 'D']:
                        option_letters.append({
                            'letter': text_clean.upper(),
                            'x': x, 'y': y, 'w': width, 'h': height
                        })
            except:
                pass
        
        return questions, option_letters

    def estimate_answer_positionally(self, q_num, q_data, filled_bubbles, image):
        """Estimate answer based on position when OCR failed to detect options"""
        h, w = image.shape[:2]
        
        # Calculate expected positions for options A, B, C, D
        q_x, q_y = q_data['x'], q_data['y']
        
        # Estimate option positions based on typical OMR sheet layout
        option_spacing = w * 0.04  # 4% of image width between options
        first_option_x = q_x + w * 0.08  # Start ~8% of width after question number
        
        estimated_positions = {
            'A': (first_option_x, q_y),
            'B': (first_option_x + option_spacing, q_y),
            'C': (first_option_x + 2*option_spacing, q_y),
            'D': (first_option_x + 3*option_spacing, q_y)
        }
        
        marked_options = []
        
        for opt_letter, (opt_x, opt_y) in estimated_positions.items():
            # Look for filled bubbles near estimated position
            best_bubble = None
            min_distance = float('inf')
            
            for bubble in filled_bubbles:
                dx = bubble['x'] - opt_x
                dy = bubble['y'] - (opt_y + 25)
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 60 and distance < min_distance:
                    min_distance = distance
                    best_bubble = bubble
            
            if best_bubble:
                marked_options.append(opt_letter)
        
        # Return result
        if len(marked_options) == 0:
            return "NONE"
        elif len(marked_options) == 1:
            return marked_options[0]
        else:
            return "MULTIPLE"

    def detect_bubbles_hybrid(self, image: np.ndarray) -> Dict[str, str]:
        """Hybrid approach: Use visual pattern recognition for sheet structure"""
        print("Using hybrid visual pattern detection...")
        
        # For now, use the enhanced grid approach but with visual bubble detection
        answers = {}
        
        # Find all actual filled bubbles first
        filled_bubbles = self.find_filled_bubbles(image)
        
        # Use visual analysis to map bubbles to grid positions
        # This is a simplified approach - in a real implementation, we'd use more
        # sophisticated computer vision techniques
        
        h, w = image.shape[:2]
        
        # Estimate grid layout based on image analysis
        # These are rough estimates that would need calibration
        questions_per_col = 20
        num_cols = 5
        
        for col in range(num_cols):
            for row in range(questions_per_col):
                q_num = col * questions_per_col + row + 1
                
                # Estimate where this question's bubbles should be
                # This is very rough and would need proper calibration
                base_x = w * 0.15 + col * (w * 0.17)  # Proportional positioning
                base_y = h * 0.15 + row * (h * 0.04)  # Proportional positioning
                
                marked_options = []
                
                # Check each option position (A, B, C, D)
                for choice in range(4):
                    expected_x = base_x + choice * (w * 0.03)
                    expected_y = base_y
                    
                    # Find nearest filled bubble
                    nearest_bubble = None
                    min_dist = float('inf')
                    
                    for bubble in filled_bubbles:
                        dx = bubble['x'] - expected_x
                        dy = bubble['y'] - expected_y
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        # Use proportional distance threshold
                        threshold = min(w, h) * 0.05  # 5% of image dimension
                        
                        if dist < threshold and dist < min_dist:
                            min_dist = dist
                            nearest_bubble = bubble
                    
                    if nearest_bubble:
                        marked_options.append(chr(ord('A') + choice))
                
                # Determine answer
                if len(marked_options) == 0:
                    answers[str(q_num)] = "NONE"
                elif len(marked_options) == 1:
                    answers[str(q_num)] = marked_options[0]
                else:
                    answers[str(q_num)] = "MULTIPLE"
        
        return answers

    def find_filled_bubbles(self, image: np.ndarray):
        """Find all filled bubbles in the image using contour analysis"""
        mask = self.create_bubble_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_bubbles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.BUBBLE_MIN_AREA <= area <= self.config.BUBBLE_MAX_AREA:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        
                        # Calculate fill ratio
                        roi_mask = np.zeros(mask.shape, dtype=np.uint8)
                        cv2.circle(roi_mask, (int(x), int(y)), int(radius), 255, -1)
                        bubble_region = cv2.bitwise_and(mask, mask, mask=roi_mask)
                        fill_ratio = cv2.countNonZero(bubble_region) / (np.pi * radius * radius)
                        
                        if fill_ratio > self.config.MARKING_THRESHOLD:
                            filled_bubbles.append({
                                'x': int(x), 'y': int(y), 
                                'radius': radius, 'fill_ratio': fill_ratio
                            })
        
        return filled_bubbles

    def detect_bubbles_grid(self, image: np.ndarray) -> Dict[str, str]:
        """Fallback: try OCR-based first, then grid if OCR fails"""
        try:
            return self.detect_bubbles_ocr_based(image)
        except Exception as e:
            print(f"OCR detection failed: {e}, using basic grid fallback")
            # Minimal grid fallback for emergency cases
            mask = self.create_bubble_mask(image)
            answers = {}
            for q_num in range(1, 101):
                answers[str(q_num)] = "NONE"  # Conservative approach
            return answers

# ==================== ANSWER KEY MANAGER ====================

class AnswerKeyManager:
    """Manages answer keys from Excel files"""
    
    @staticmethod
    def load_from_excel(file_path: Path) -> Optional[Dict]:
        """Load answer keys from Excel file"""
        try:
            xls = pd.ExcelFile(file_path)
            answer_keys = {}
            
            # Try sheet-based format first
            for sheet_name in xls.sheet_names:
                if 'set' in sheet_name.lower():
                    df = pd.read_excel(xls, sheet_name, header=None)
                    key = AnswerKeyManager._parse_answer_column(df)
                    if key:
                        set_label = 'Set_A' if 'a' in sheet_name.lower() else 'Set_B'
                        answer_keys[set_label] = key
            
            # If no sheet-based keys, try column-based
            if not answer_keys:
                df = pd.read_excel(xls, xls.sheet_names[0])
                for col in df.columns:
                    key = AnswerKeyManager._parse_answer_column(df[col])
                    if key:
                        answer_keys[col] = key
            
            return answer_keys if answer_keys else None
            
        except Exception as e:
            print(f"Error loading answer keys: {e}")
            return None
    
    @staticmethod
    def _parse_answer_column(data) -> Dict[str, List[str]]:
        """Parse answer data from column or dataframe"""
        answers = {}
        
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]  # Take first column
        
        for item in data:
            if pd.isna(item):
                continue
            
            item_str = str(item).strip()
            # Parse format: "1-a" or "1 - a" or "1. a"
            match = re.match(r'(\d+)[\s\-\.]+([a-dA-D,\s]+)', item_str)
            if match:
                q_num = match.group(1)
                options = [opt.strip().lower() for opt in match.group(2).split(',')]
                answers[q_num] = options
        
        return answers

# ==================== REPORT GENERATOR ====================

class ReportGenerator:
    """Generates evaluation reports"""
    
    def __init__(self):
        self.config = OMRConfig()
    
    def generate_report(self, image_path: str, student_answers: Dict, 
                       answer_key: Dict, exam_set: str, 
                       subjects: List[str] = None) -> Dict:
        """Generate comprehensive evaluation report with multi-mark handling"""
        
        # Initialize scores
        total_score = 0
        subject_scores = {}
        
        if subjects:
            subject_scores = {subj: 0 for subj in subjects}
        else:
            subject_scores = {f"Subject_{i+1}": 0 for i in range(self.config.COLUMNS_PER_SHEET)}
        
        # Evaluation summary
        summary = {
            "correct": 0,
            "incorrect": 0,
            "unattempted": 0,
            "multiple_marked": 0,
            "partially_marked": 0
        }
        
        # Detailed evaluation
        details = []
        
        for q_num in range(1, self.config.TOTAL_QUESTIONS + 1):
            q_str = str(q_num)
            correct_answers = answer_key.get(q_str, [])
            
            # Handle empty answer arrays (common issue)
            if not correct_answers or (isinstance(correct_answers, list) and len(correct_answers) == 0):
                # Skip questions with no answer key
                continue
                
            if isinstance(correct_answers, str):
                correct_answers = [correct_answers.lower()]
            else:
                correct_answers = [ans.lower() for ans in correct_answers if ans]  # Filter empty strings
            
            student_answer = student_answers.get(q_str, "NONE")
            
            # Handle different answer states
            if student_answer == "NONE":
                status = "Unattempted"
                summary["unattempted"] += 1
            elif student_answer == "MULTIPLE":
                status = "Multiple Marked"
                summary["multiple_marked"] += 1
            elif student_answer == "PARTIAL":
                status = "Partially Marked"
                summary["partially_marked"] += 1
            elif student_answer.lower() in correct_answers:
                status = "Correct"
                summary["correct"] += 1
                total_score += 1
                
                # Update subject score
                subject_idx = (q_num - 1) // self.config.QUESTIONS_PER_COLUMN
                if subjects and subject_idx < len(subjects):
                    subject_scores[subjects[subject_idx]] += 1
                else:
                    subject_scores[f"Subject_{subject_idx+1}"] += 1
            else:
                status = "Incorrect"
                summary["incorrect"] += 1
            
            details.append({
                "question": q_num,
                "marked": student_answer,
                "correct": correct_answers,
                "status": status
            })
        
        # Calculate total questions attempted (excluding empty answer keys)
        total_questions_with_keys = len([q for q in range(1, self.config.TOTAL_QUESTIONS + 1) 
                                       if answer_key.get(str(q)) and 
                                       len(answer_key.get(str(q), [])) > 0])
        
        # Generate report
        report = {
            "source_image": os.path.basename(image_path),
            "exam_set": exam_set,
            "total_score": total_score,
            "max_score": total_questions_with_keys,
            "percentage": round(total_score * 100 / total_questions_with_keys, 2) if total_questions_with_keys > 0 else 0,
            "subject_scores": subject_scores,
            "summary": summary,
            "details": details,
            "timestamp": pd.Timestamp.now().isoformat(),
            "questions_with_answer_key": total_questions_with_keys
        }
        
        return report

# ==================== MAIN OMR PROCESSOR ====================

class OMRProcessor:
    """Main processor orchestrating the entire pipeline"""
    
    def __init__(self):
        self.dataset_mgr = DatasetManager()
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine()
        self.bubble_detector = BubbleDetector()
        self.report_gen = ReportGenerator()
        self.answer_keys = None
        
        # Load answer keys
        if self.dataset_mgr.key_file:
            self.answer_keys = AnswerKeyManager.load_from_excel(self.dataset_mgr.key_file)
            if self.answer_keys:
                # Dynamic choice count: scan keys for max option letter
                max_letter = 'D'
                for key_map in self.answer_keys.values():
                    for ans_list in key_map.values():
                        if isinstance(ans_list, list):
                            for opt in ans_list:
                                if opt and opt[0].isalpha():
                                    letter = opt[0].upper()
                                    if letter > max_letter:
                                        max_letter = letter
                        elif isinstance(ans_list, str) and ans_list:
                            letter = ans_list[0].upper()
                            if letter > max_letter:
                                max_letter = letter
                choice_count = ord(max_letter) - ord('A') + 1
                if choice_count != OMRConfig.CHOICES_PER_QUESTION:
                    print(f"[INFO] Adjusting CHOICES_PER_QUESTION from {OMRConfig.CHOICES_PER_QUESTION} to {choice_count}")
                    OMRConfig.CHOICES_PER_QUESTION = choice_count
        
        # Create output directory
        self.dataset_mgr.output_dir.mkdir(exist_ok=True)
    
    def process_image(self, image_path: Path, expected_set: str = None) -> Optional[Dict]:
        """Process single OMR sheet image"""
        print(f"Processing: {image_path.name}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        # Try to detect exam set
        detected_set = self.ocr_engine.extract_set_code(image)
        exam_set = f"Set_{detected_set}" if detected_set else expected_set
        
        # Transform perspective
        warped = self.preprocessor.transform_perspective(image)
        fallback_mode = False
        if warped is None:
            print(f"Failed to find OMR sheet in: {image_path.name}")
            return None
        # Heuristic: if contour not found original aspect maybe far from target; detect by size mismatch typical? Already resized above.
        if warped.shape[0] == 1200 and warped.shape[1] == 800 and 'Set' not in str(image_path):
            # Not robust; just treat resize path as potential fallback when no subject blocks found later.
            fallback_mode = True
        
        # Detect subjects (dynamic OCR)
        subject_blocks = self.ocr_engine.detect_subjects(warped)
        subject_names = list(subject_blocks.keys()) if subject_blocks else None
        
        # Detect question structure if subjects found
        student_answers = {}
        
        if subject_blocks:
            # OCR-based detection
            structure = self.ocr_engine.find_question_structure(warped, subject_blocks)
            
            for q_num, q_data in structure.items():
                marked = self.bubble_detector.detect_marked_bubble(warped, q_data['options'])
                student_answers[q_num] = marked if marked else "NONE"
        
        # Fallback to grid-based detection if needed
        if not student_answers:
            print("Using grid-based bubble detection...")
            student_answers = self.bubble_detector.detect_bubbles_grid(warped)
        
        # Get answer key
        answer_key = {}
        if self.answer_keys:
            if exam_set in self.answer_keys:
                answer_key = self.answer_keys[exam_set]
            elif 'GLOBAL' in self.answer_keys:
                answer_key = self.answer_keys['GLOBAL']
            else:
                # Use first available key
                answer_key = list(self.answer_keys.values())[0]
        
        # Generate report
        report = self.report_gen.generate_report(
            str(image_path),
            student_answers,
            answer_key,
            exam_set,
            subject_names
        )
        report['perspective_fallback'] = fallback_mode
        
        # Add metadata
        report['detected_set_code'] = detected_set
        report['subjects_detected'] = subject_names is not None
        report['detection_method'] = 'OCR-based' if subject_blocks else 'Grid-based'
        
        # Advanced debug visualization for OCR-based detection
        if OMRConfig.DEBUG and not hasattr(self, '_debug_done'):
            debug_img = warped.copy()
            
            # Show OCR-detected text positions
            try:
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                
                # Draw detected text
                for i, text in enumerate(ocr_data['text']):
                    if not text or not text.strip():
                        continue
                    
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i] 
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    text_clean = text.strip()
                    
                    # Question numbers in blue
                    if text_clean.isdigit() and 1 <= int(text_clean) <= 100:
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(debug_img, f"Q{text_clean}", (x, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    # Option letters in green
                    elif text_clean.upper() in ['A', 'B', 'C', 'D']:
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(debug_img, text_clean.upper(), (x, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Show detected filled bubbles in red
                mask = self.bubble_detector.create_bubble_mask(warped)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                bubble_count = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if OMRConfig.BUBBLE_MIN_AREA <= area <= OMRConfig.BUBBLE_MAX_AREA:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3:
                                (x, y), radius = cv2.minEnclosingCircle(cnt)
                                
                                # Calculate fill ratio
                                roi_mask = np.zeros(mask.shape, dtype=np.uint8)
                                cv2.circle(roi_mask, (int(x), int(y)), int(radius), 255, -1)
                                bubble_region = cv2.bitwise_and(mask, mask, mask=roi_mask)
                                fill_ratio = cv2.countNonZero(bubble_region) / (np.pi * radius * radius)
                                
                                if fill_ratio > OMRConfig.MARKING_THRESHOLD:
                                    # Red circles for filled bubbles
                                    cv2.circle(debug_img, (int(x), int(y)), int(radius), (0, 0, 255), 3)
                                    cv2.putText(debug_img, f"{int(fill_ratio*100)}%", 
                                              (int(x)+15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.4, (0, 0, 255), 1)
                                    bubble_count += 1
                
                # Add legend
                cv2.putText(debug_img, f"Blue=Questions, Green=Options, Red=Filled({bubble_count})", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Debug visualization error: {e}")
                # Fallback simple visualization
                cv2.putText(debug_img, "OCR-based detection active", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite(str(self.dataset_mgr.output_dir / 'debug_ocr_analysis.jpg'), debug_img)
            print(f"OCR Debug visualization saved: debug_ocr_analysis.jpg")
            self._debug_done = True
        
        return report
    
    def process_pdf(self, pdf_path: Path, expected_set: str = None) -> List[Dict]:
        """Process PDF with multiple OMR sheets"""
        reports = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                
                # Convert to numpy array
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                img_data = img_data.reshape(pix.h, pix.w, pix.n)
                
                if pix.n == 4:  # RGBA
                    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                
                # Save temporary image
                temp_path = self.dataset_mgr.output_dir / f"temp_{pdf_path.stem}_page_{page_num+1}.png"
                cv2.imwrite(str(temp_path), img_bgr)
                
                # Process image
                report = self.process_image(temp_path, expected_set)
                if report:
                    report['source_page'] = page_num + 1
                    reports.append(report)
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
        
        return reports
    
    def run(self):
        """Main execution loop"""
        print("=" * 60)
        print("AUTOMATED OMR EVALUATION SYSTEM")
        print("=" * 60)
        
        if not self.answer_keys:
            print("ERROR: No answer keys loaded!")
            return
        
        print(f"Dataset root: {self.dataset_mgr.root}")
        print(f"Answer keys loaded: {list(self.answer_keys.keys())}")
        print(f"Output directory: {self.dataset_mgr.output_dir}")
        print()
        
        total_processed = 0
        total_reports = 0
        
        # Process each exam set
        for set_name, set_folder in self.dataset_mgr.sets.items():
            print(f"\nProcessing {set_name} from {set_folder.name}/")
            print("-" * 40)
            
            # Process all images and PDFs in folder
            for file_path in set_folder.iterdir():
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    report = self.process_image(file_path, set_name)
                    if report:
                        # Save report
                        output_path = self.dataset_mgr.output_dir / f"{file_path.stem}_report.json"
                        with open(output_path, 'w') as f:
                            json.dump(report, f, indent=4)
                        print(f"  ✓ {file_path.name} -> Score: {report['total_score']}/{report['max_score']}")
                        total_reports += 1
                    total_processed += 1
                    
                elif file_path.suffix.lower() == '.pdf':
                    reports = self.process_pdf(file_path, set_name)
                    for i, report in enumerate(reports):
                        output_path = self.dataset_mgr.output_dir / f"{file_path.stem}_page{i+1}_report.json"
                        with open(output_path, 'w') as f:
                            json.dump(report, f, indent=4)
                        print(f"  ✓ {file_path.name} (Page {i+1}) -> Score: {report['total_score']}/{report['max_score']}")
                        total_reports += 1
                    total_processed += len(reports) if reports else 1
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n" + "=" * 60)
        print(f"PROCESSING COMPLETE")
        print(f"Total files processed: {total_processed}")
        print(f"Total reports generated: {total_reports}")
        print(f"Results saved to: {self.dataset_mgr.output_dir}")
        print("=" * 60)
    
    def generate_summary_report(self):
        """Generate overall summary of all evaluations"""
        all_reports = []
        
        # Load all JSON reports
        for json_file in self.dataset_mgr.output_dir.glob("*.json"):
            if 'summary' not in json_file.name:
                with open(json_file, 'r') as f:
                    all_reports.append(json.load(f))
        
        if not all_reports:
            return
        
        # Calculate statistics
        summary = {
            "total_sheets_evaluated": len(all_reports),
            "evaluation_date": pd.Timestamp.now().isoformat(),
            "statistics": {
                "average_score": round(sum(r['total_score'] for r in all_reports) / len(all_reports), 2),
                "highest_score": max(r['total_score'] for r in all_reports),
                "lowest_score": min(r['total_score'] for r in all_reports),
                "pass_percentage": round(sum(1 for r in all_reports if r['percentage'] >= 60) * 100 / len(all_reports), 2)
            },
            "subject_wise_average": {},
            "set_wise_stats": {}
        }
        
        # Subject-wise analysis
        all_subjects = set()
        for report in all_reports:
            all_subjects.update(report.get('subject_scores', {}).keys())
        
        for subject in all_subjects:
            scores = []
            for report in all_reports:
                if subject in report.get('subject_scores', {}):
                    scores.append(report['subject_scores'][subject])
            if scores:
                summary['subject_wise_average'][subject] = round(sum(scores) / len(scores), 2)
        
        # Set-wise analysis
        for set_name in ['Set_A', 'Set_B']:
            set_reports = [r for r in all_reports if r.get('exam_set') == set_name]
            if set_reports:
                summary['set_wise_stats'][set_name] = {
                    "count": len(set_reports),
                    "average": round(sum(r['total_score'] for r in set_reports) / len(set_reports), 2)
                }
        
        # Save summary
        summary_path = self.dataset_mgr.output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Generate CSV report for easy viewing
        self.generate_csv_report(all_reports)
    
    def generate_csv_report(self, reports: List[Dict]):
        """Generate CSV report for spreadsheet viewing"""
        rows = []
        
        for report in reports:
            row = {
                'Image': report['source_image'],
                'Set': report['exam_set'],
                'Total Score': report['total_score'],
                'Percentage': report['percentage'],
                'Correct': report['summary']['correct'],
                'Incorrect': report['summary']['incorrect'],
                'Unattempted': report['summary']['unattempted'],
                'Detection Method': report.get('detection_method', 'Unknown')
            }
            
            # Add subject scores
            for subject, score in report.get('subject_scores', {}).items():
                row[f'{subject} Score'] = score
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        csv_path = self.dataset_mgr.output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\nCSV report saved to: {csv_path}")

# ==================== DEBUGGING UTILITIES ====================

class DebugVisualizer:
    """Utilities for debugging and visualization"""
    
    @staticmethod
    def visualize_bubble_detection(image: np.ndarray, answers: Dict, output_path: Path):
        """Visualize detected bubbles on the image"""
        vis_image = image.copy()
        
        # Draw detected answers
        for q_num, answer in answers.items():
            if answer != "NONE":
                # Calculate approximate position (needs calibration)
                q_int = int(q_num)
                col = (q_int - 1) // 20
                row = (q_int - 1) % 20
                
                # Draw marker
                x = 150 + col * 140 + (ord(answer) - ord('A')) * 35
                y = 300 + row * 30
                
                cv2.circle(vis_image, (x, y), 15, (0, 255, 0), 2)
                cv2.putText(vis_image, f"Q{q_num}:{answer}", 
                          (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.4, (0, 255, 0), 1)
        
        cv2.imwrite(str(output_path), vis_image)
    
    @staticmethod
    def save_processing_stages(image: np.ndarray, output_dir: Path, prefix: str):
        """Save intermediate processing stages for debugging"""
        # Create debug subdirectory
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        # Original
        cv2.imwrite(str(debug_dir / f"{prefix}_1_original.jpg"), image)
        
        # Enhanced
        preprocessor = ImagePreprocessor()
        enhanced = preprocessor.enhance_image(image)
        cv2.imwrite(str(debug_dir / f"{prefix}_2_enhanced.jpg"), enhanced)
        
        # Bubble mask
        detector = BubbleDetector()
        mask = detector.create_bubble_mask(image)
        cv2.imwrite(str(debug_dir / f"{prefix}_3_bubble_mask.jpg"), mask)

# ==================== MAIN EXECUTION ====================

def main():
    """Main entry point"""
    try:
        # Check for required packages
        required = ['cv2', 'numpy', 'pandas', 'fitz']
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            print(f"Missing required packages: {missing}")
            print("Install with: pip install opencv-python numpy pandas PyMuPDF")
            if 'pytesseract' not in globals():
                print("Optional: pip install pytesseract (for OCR features)")
            return
        
        # Run processor
        processor = OMRProcessor()
        processor.run()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()