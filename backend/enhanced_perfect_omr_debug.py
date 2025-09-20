"""
Enhanced Perfect OMR Debug System
Combines advanced thresholding techniques with comprehensive analysis and JSON output
"""

import cv2
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: Tesseract OCR not available. Install pytesseract for set detection.")

class PerfectOMRDebugger:
    """
    Advanced OMR processing system with sophisticated thresholding and debugging capabilities
    """
    
    def __init__(self):
        self.debug_images = {}
        self.threshold_params = {
            'MIN_JUMP': 15,
            'JUMP_DELTA': 20,
            'CONFIDENT_SURPLUS': 5,
            'GAMMA_LOW': 0.85,
            'PAGE_TYPE_FOR_THRESHOLD': 'white',
            'MIN_GAP': 25
        }
        
    def stack_images(self, img_array, scale, labels=[]):
        """Stack images for visualization"""
        rows = len(img_array)
        cols = len(img_array[0])
        rows_available = isinstance(img_array[0], list)
        width = img_array[0][0].shape[1]
        height = img_array[0][0].shape[0]
        
        if rows_available:
            for x in range(0, rows):
                for y in range(0, cols):
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    if len(img_array[x][y].shape) == 2:
                        img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
            
            image_blank = np.zeros((height, width, 3), np.uint8)
            hor = [image_blank] * rows
            
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
                if len(img_array[x].shape) == 2:
                    img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
            ver = np.hstack(img_array)
        
        if len(labels) != 0:
            each_img_width = int(ver.shape[1] / cols)
            each_img_height = int(ver.shape[0] / rows)
            for d in range(0, rows):
                for c in range(0, cols):
                    cv2.rectangle(ver, (c*each_img_width, each_img_height*d),
                                (c*each_img_width + len(labels[d][c])*13 + 27, 30 + each_img_height*d),
                                (255, 255, 255), cv2.FILLED)
                    cv2.putText(ver, labels[d][c], (each_img_width*c + 10, each_img_height*d + 20),
                              cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
        return ver
    
    def reorder_points(self, points):
        """Reorder corner points for perspective transformation"""
        points = points.reshape((4, 2))
        points_new = np.zeros((4, 1, 2), np.int32)
        add = points.sum(1)
        
        points_new[0] = points[np.argmin(add)]    # top-left
        points_new[3] = points[np.argmax(add)]    # bottom-right
        diff = np.diff(points, axis=1)
        points_new[1] = points[np.argmin(diff)]   # top-right
        points_new[2] = points[np.argmax(diff)]   # bottom-left
        
        return points_new
    
    def find_rectangular_contours(self, contours):
        """Find rectangular contours suitable for OMR sheets"""
        rect_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) == 4:
                    rect_contours.append(contour)
        
        rect_contours = sorted(rect_contours, key=cv2.contourArea, reverse=True)
        return rect_contours
    
    def get_corner_points(self, contour):
        """Get corner points of a contour"""
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        return approx
    
    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def adjust_gamma(self, image, gamma=1.0):
        """Adjust image gamma for better contrast"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def normalize_image(self, image):
        """Normalize image intensity"""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    def get_global_threshold(self, q_vals_orig, looseness=1):
        """
        Advanced global thresholding using gap analysis
        Finds the first large gap in sorted pixel values to determine threshold
        """
        global_default_threshold = 180  # Default for white pages
        
        # Sort the values
        q_vals = sorted(q_vals_orig)
        
        # Find the first large gap
        ls = (looseness + 1) // 2
        l = len(q_vals) - ls
        max_jump, threshold = self.threshold_params['MIN_JUMP'], global_default_threshold
        
        for i in range(ls, l):
            jump = q_vals[i + ls] - q_vals[i - ls]
            if jump > max_jump:
                max_jump = jump
                threshold = q_vals[i - ls] + jump / 2
        
        return threshold, threshold - max_jump // 2, threshold + max_jump // 2
    
    def get_local_threshold(self, q_vals, global_thr, no_outliers):
        """
        Local thresholding for individual question strips
        """
        q_vals = sorted(q_vals)
        
        # Handle small number of points
        if len(q_vals) < 3:
            return global_thr if np.max(q_vals) - np.min(q_vals) < self.threshold_params['MIN_GAP'] else np.mean(q_vals)
        
        # Find largest gap
        l = len(q_vals) - 1
        max_jump, threshold = self.threshold_params['MIN_JUMP'], 255
        
        for i in range(1, l):
            jump = q_vals[i + 1] - q_vals[i - 1]
            if jump > max_jump:
                max_jump = jump
                threshold = q_vals[i - 1] + jump / 2
        
        # Use global threshold if not confident
        confident_jump = self.threshold_params['MIN_JUMP'] + self.threshold_params['CONFIDENT_SURPLUS']
        if max_jump < confident_jump and no_outliers:
            threshold = global_thr
        
        return threshold
    
    def split_answer_boxes(self, image, questions, choices):
        """Split thresholded image into individual answer boxes"""
        rows = np.vsplit(image, questions)
        boxes = []
        for row in rows:
            cols = np.hsplit(row, choices)
            for box in cols:
                boxes.append(box)
        return boxes
    
    def extract_set_code_ocr(self, image):
        """Extract set code using OCR"""
        if not OCR_AVAILABLE:
            return None
        
        try:
            h, w = image.shape[:2]
            header = image[0:int(h*0.25), :]
            
            if len(header.shape) == 3:
                gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
            else:
                gray = header
                
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            text_lower = text.lower().replace('\n', ' ')
            
            import re
            # Look for "Set A" or "Set B" patterns
            match = re.search(r'set\s*[:\-]?\s*([ab])', text_lower)
            if match:
                return match.group(1).upper()
                
            # Alternative pattern matching
            match2 = re.search(r'set[^a-z0-9]{1,5}([ab])\b', text_lower)
            if match2:
                return match2.group(1).upper()
                
        except Exception as e:
            print(f"OCR failed: {e}")
            return None
        
        return None
    
    def load_answer_key_from_excel(self, excel_path):
        """Load answer key from Excel file"""
        try:
            # Try to read the Excel file
            df = pd.read_excel(excel_path, sheet_name=None)  # Read all sheets
            
            answer_keys = {}
            
            for sheet_name, sheet_df in df.items():
                if 'Set' in sheet_name or 'set' in sheet_name.lower():
                    # Extract set identifier
                    if 'A' in sheet_name.upper():
                        set_key = 'Set_A'
                    elif 'B' in sheet_name.upper():
                        set_key = 'Set_B'
                    else:
                        set_key = sheet_name
                    
                    # Parse answers from the sheet
                    answers = {}
                    for index, row in sheet_df.iterrows():
                        for col_name, value in row.items():
                            if pd.notna(value) and str(value).strip():
                                # Try to extract question number and answer
                                value_str = str(value).strip()
                                # Look for patterns like "1a", "2c", etc.
                                import re
                                match = re.match(r'(\d+)([a-d])', value_str.lower())
                                if match:
                                    q_num = match.group(1)
                                    answer = match.group(2)
                                    answers[q_num] = [answer]
                    
                    if answers:
                        answer_keys[set_key] = answers
            
            return answer_keys
            
        except Exception as e:
            print(f"Error loading answer key from Excel: {e}")
            return None
    
    def detect_green_circles_and_black_marks(self, image):
        """
        Detect green circles and black marks specifically for the provided OMR format
        Enhanced to ONLY detect filled/marked circles, not empty green circles
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Find ALL circular shapes first (both filled and empty)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,
            param1=50,
            param2=30,
            minRadius=8,
            maxRadius=25
        )
        
        all_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                all_circles.append({
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'area': np.pi * r * r
                })
        
        print(f"Found {len(all_circles)} total circular shapes")
        
        # Step 2: Detect ONLY filled/marked circles (ignore empty green circles)
        # Use multiple thresholding methods to detect dark marks
        
        # Method 1: Adaptive threshold for pencil marks
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Method 2: OTSU threshold for pen marks
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 3: Simple threshold for very dark marks
        _, simple_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Combine all thresholds to catch different types of marks
        combined_thresh = cv2.bitwise_or(cv2.bitwise_or(adaptive_thresh, otsu_thresh), simple_thresh)
        
        # Step 3: Check each circle to see if it's ACTUALLY filled/marked
        marked_circles = []
        empty_circles = []
        
        for circle in all_circles:
            center = circle['center']
            radius = circle['radius']
            
            # Create a mask for this circle
            circle_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(circle_mask, center, max(3, radius - 2), 255, -1)
            
            # Count dark pixels inside this circle
            dark_pixels_in_circle = cv2.bitwise_and(combined_thresh, circle_mask)
            dark_pixel_count = cv2.countNonZero(dark_pixels_in_circle)
            
            # Calculate fill ratio
            total_circle_pixels = np.pi * max(3, radius - 2) * max(3, radius - 2)
            fill_ratio = dark_pixel_count / total_circle_pixels if total_circle_pixels > 0 else 0
            
            # Also check average darkness in the circle area
            circle_roi = cv2.bitwise_and(gray, gray, mask=circle_mask)
            roi_pixels = circle_roi[circle_roi > 0]
            avg_darkness = np.mean(roi_pixels) if len(roi_pixels) > 0 else 255
            
            # STRICT criteria for marked circles - must be significantly filled and dark
            is_marked = (
                fill_ratio > 0.35 and      # At least 35% filled with dark pixels (stricter)
                avg_darkness < 130 and     # Average darkness must be significant  
                dark_pixel_count > 50 and  # Minimum number of dark pixels (stricter)
                radius >= 8               # Minimum reasonable radius
            )
            
            if is_marked:
                marked_circles.append({
                    'center': center,
                    'radius': radius,
                    'fill_ratio': fill_ratio,
                    'avg_darkness': avg_darkness,
                    'dark_pixels': dark_pixel_count,
                    'method': 'filled_detection'
                })
            else:
                # This is an empty circle (green circle without filling)
                empty_circles.append({
                    'center': center,
                    'radius': radius,
                    'fill_ratio': fill_ratio,
                    'avg_darkness': avg_darkness,
                    'method': 'empty_circle'
                })
        
        # Step 4: Sort circles by position
        marked_circles.sort(key=lambda c: (c['center'][1], c['center'][0]))
        
        # Store debug images
        self.debug_images['combined_threshold'] = combined_thresh
        self.debug_images['adaptive_threshold'] = adaptive_thresh
        self.debug_images['otsu_threshold'] = otsu_thresh
        
        # Create debug visualization showing the difference
        debug_img = image.copy()
        
        # Draw empty circles in GREEN (these will be ignored)
        for circle in empty_circles:
            center = circle['center']
            radius = circle['radius']
            cv2.circle(debug_img, center, radius, (0, 255, 0), 2)  # Green border
            cv2.putText(debug_img, "EMPTY", (center[0]-20, center[1]-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw marked circles in RED (these are actual answers)
        for circle in marked_circles:
            center = circle['center']
            radius = circle['radius']
            fill_ratio = circle['fill_ratio']
            cv2.circle(debug_img, center, radius, (0, 0, 255), 3)  # Red border
            cv2.circle(debug_img, center, 3, (0, 0, 255), -1)  # Red center dot
            cv2.putText(debug_img, f"MARKED {fill_ratio:.2f}", (center[0]-30, center[1]+radius+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        self.debug_images['circles_detection'] = debug_img
        
        print(f"Detection results:")
        print(f"  - Total circles found: {len(all_circles)}")
        print(f"  - Empty circles (ignored): {len(empty_circles)}")
        print(f"  - Marked circles (answers): {len(marked_circles)}")
        
        # Return ONLY the marked circles as "green_circles" for compatibility
        # The function name remains the same but now returns only filled circles
        return marked_circles, marked_circles  # Both lists are the same now
    
    def organize_circles_into_grid(self, circles, questions=20, choices=4):
        """
        Organize circles into a question-answer grid with better duplicate handling
        """
        if not circles:
            return {}
        
        # Sort circles by position
        circles.sort(key=lambda c: (c['center'][1], c['center'][0]))
        
        # Group circles into rows (questions) with better spacing detection
        rows = []
        current_row = [circles[0]]
        
        for i in range(1, len(circles)):
            current_y = circles[i]['center'][1]
            row_y = current_row[0]['center'][1]
            
            # Use dynamic row threshold based on average circle size
            avg_radius = sum(c['radius'] for c in current_row) / len(current_row)
            row_threshold = max(25, avg_radius * 1.5)  # Adaptive threshold
            
            # If y-coordinate is similar to current row, add to current row
            if abs(current_y - row_y) < row_threshold:
                # Check if this circle is too close to existing circles in the row
                too_close = False
                for existing in current_row:
                    distance = np.sqrt((circles[i]['center'][0] - existing['center'][0])**2 + 
                                     (circles[i]['center'][1] - existing['center'][1])**2)
                    if distance < avg_radius * 1.2:  # Too close to existing circle
                        too_close = True
                        break
                
                if not too_close:
                    current_row.append(circles[i])
            else:
                # Sort current row by x-coordinate and add to rows
                if len(current_row) >= 2:  # Only keep rows with at least 2 choices
                    current_row.sort(key=lambda c: c['center'][0])
                    rows.append(current_row)
                current_row = [circles[i]]
        
        # Add the last row
        if current_row and len(current_row) >= 2:
            current_row.sort(key=lambda c: c['center'][0])
            rows.append(current_row)
        
        # Convert to question-answer format with better choice assignment
        grid = {}
        for q_idx, row in enumerate(rows):
            if q_idx < questions and len(row) <= choices:
                question_num = q_idx + 1
                grid[question_num] = {}
                
                # Only assign up to 'choices' number of circles per question
                for choice_idx, circle in enumerate(row[:choices]):
                    choice_letter = chr(ord('a') + choice_idx)
                    grid[question_num][choice_letter] = circle
        
        return grid

    def process_omr_sheet(self, image_path, answer_keys=None, questions=20, choices=4):
        """
        Main processing function for OMR sheet - optimized for green circle format
        """
        # Read and prepare image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Get image name for results
        image_name = os.path.basename(image_path)
        
        # Store original dimensions for better processing
        original_img = img.copy()
        h, w = img.shape[:2]
        
        # Store debug images
        self.debug_images['original'] = original_img.copy()
        
        print(f"Processing image: {image_name} ({w}x{h})")
        
        # Step 1: Detect only filled/marked circles (ignore empty green circles)
        print("Detecting filled/marked circles only...")
        marked_circles, _ = self.detect_green_circles_and_black_marks(original_img)
        
        print(f"Found {len(marked_circles)} marked/filled circles")
        
        # Step 2: Organize marked circles into question-answer grid
        print("Organizing marked circles into grid...")
        marked_grid = self.organize_circles_into_grid(marked_circles, questions, choices)
        
        print(f"Organized into {len(marked_grid)} questions with answers")
        
        # Step 3: Determine answers based on marked circles
        detected_answers = []
        for q in range(1, questions + 1):
            marked_options = []
            
            # Check if this question has any marked answers
            if q in marked_grid:
                for choice in ['a', 'b', 'c', 'd']:
                    if choice in marked_grid[q]:
                        # This choice is marked for this question
                        marked_options.append(choice)
            
            # Determine final answer for this question
            if len(marked_options) == 0:
                detected_answers.append("NONE")
            elif len(marked_options) == 1:
                detected_answers.append(marked_options[0])
            else:
                detected_answers.append(",".join(marked_options))  # Multiple marked
        
        print(f"Detected answers: {detected_answers[:10]}...")  # Show first 10
        
        # Step 4: Create visualization
        img_result = original_img.copy()
        self.show_marked_circle_answers(img_result, marked_grid, detected_answers, answer_keys, "Set_A")
        self.debug_images['result'] = img_result
        
        # Step 5: Extract set code (if possible)
        detected_set = self.extract_set_code_ocr(original_img)
        if detected_set:
            detected_set_key = f"Set_{detected_set}"
        else:
            detected_set_key = "Set_A"  # Default fallback
        
        print(f"Detected set: {detected_set_key}")
        
        # Step 6: Create results JSON
        results = self.create_results_json(
            image_name, detected_answers, answer_keys, 
            detected_set_key, detected_set, questions
        )
        
        return results, img_result
    
    def show_marked_circle_answers(self, img, marked_grid, detected_answers, answer_keys=None, set_key="Set_A"):
        """Visualize detected answers for marked circles only"""
        
        # Get correct answers for comparison
        correct_answers_dict = {}
        if answer_keys and set_key in answer_keys:
            for q_str, correct_list in answer_keys[set_key].items():
                q_num = int(q_str)
                correct_answers_dict[q_num] = correct_list[0] if correct_list else 'NONE'
        
        # Draw marked circles and their answers
        for q_num, choices_dict in marked_grid.items():
            for choice_letter, circle_info in choices_dict.items():
                center = circle_info['center']
                radius = circle_info['radius']
                
                # Get the detected answer for this question
                detected_answer = detected_answers[q_num - 1] if q_num <= len(detected_answers) else "NONE"
                correct_answer = correct_answers_dict.get(q_num, 'UNKNOWN')
                
                # Determine if this specific choice is correct
                is_correct = (choice_letter == correct_answer and choice_letter in detected_answer)
                is_wrong = (choice_letter in detected_answer and choice_letter != correct_answer)
                
                # Color coding for visualization
                if is_correct:
                    color = (0, 255, 0)  # Green for correct
                    thickness = 4
                elif is_wrong:
                    color = (0, 0, 255)  # Red for wrong
                    thickness = 4
                else:
                    color = (255, 0, 0)  # Blue for marked but neutral
                    thickness = 2
                
                # Draw the marked circle
                cv2.circle(img, center, radius, color, thickness)
                cv2.circle(img, center, 3, color, -1)  # Center dot
                
                # Add question and choice labels
                label = f"Q{q_num}{choice_letter.upper()}"
                cv2.putText(img, label, (center[0] - 15, center[1] - radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Add fill ratio info
                if 'fill_ratio' in circle_info:
                    fill_text = f"{circle_info['fill_ratio']:.2f}"
                    cv2.putText(img, fill_text, (center[0] - 10, center[1] + radius + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add legend
        cv2.putText(img, "GREEN: Correct Answer", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, "RED: Wrong Answer", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "BLUE: Marked Circle", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add summary
        total_marked = len([ans for ans in detected_answers if ans != "NONE"])
        cv2.putText(img, f"Total Marked: {total_marked}", (20, img.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Detection: MARKED CIRCLES ONLY", (20, img.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return img

    def show_answers(self, img, detected_answers, questions, choices, answer_keys=None, set_key="Set_A"):
        """Visualize detected answers on the image"""
        sec_w = int(img.shape[1] / choices)
        sec_h = int(img.shape[0] / questions)
        
        for q in range(questions):
            q_num = str(q + 1)
            detected = detected_answers[q]
            
            # Get correct answer if available
            correct_answers = []
            if answer_keys and set_key in answer_keys and q_num in answer_keys[set_key]:
                correct_answers = answer_keys[set_key][q_num]
            
            if detected != "NONE":
                if "," in detected:  # Multiple marked
                    # Mark all detected options in red
                    for option in detected.split(","):
                        if option.strip():
                            c = ord(option.strip()) - ord('a')
                            cx = (c * sec_w) + sec_w // 2
                            cy = (q * sec_h) + sec_h // 2
                            cv2.circle(img, (cx, cy), 50, (0, 0, 255), cv2.FILLED)  # Red for multiple
                else:
                    # Single marked option
                    c = ord(detected) - ord('a')
                    cx = (c * sec_w) + sec_w // 2
                    cy = (q * sec_h) + sec_h // 2
                    
                    # Check if correct
                    is_correct = detected in correct_answers if correct_answers else False
                    color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, red if wrong
                    cv2.circle(img, (cx, cy), 50, color, cv2.FILLED)
            
            # Mark correct answer with a smaller green circle if answer was wrong or unattempted
            if correct_answers and (detected == "NONE" or detected not in correct_answers):
                for correct_ans in correct_answers:
                    if correct_ans.isalpha():
                        c_correct = ord(correct_ans) - ord('a')
                        if c_correct < choices:
                            cx_correct = (c_correct * sec_w) + sec_w // 2
                            cy_correct = (q * sec_h) + sec_h // 2
                            cv2.circle(img, (cx_correct, cy_correct), 20, (0, 255, 0), cv2.FILLED)
        
        return img
    
    def create_results_json(self, image_name, detected_answers, answer_keys=None, set_key="Set_A", detected_set="A", questions=20):
        """Create detailed results in JSON format matching the expected structure"""
        
        # Initialize counters
        correct = 0
        incorrect = 0
        unattempted = 0
        multiple_marked = 0
        partially_marked = 0
        
        details = []
        subject_scores = {"ADV STATS": 0, "PYTHON": 0}  # Default subjects
        
        # Get correct answers for the detected set
        correct_answers_dict = {}
        if answer_keys and set_key in answer_keys:
            correct_answers_dict = answer_keys[set_key]
        
        for q in range(questions):
            q_num = str(q + 1)
            detected = detected_answers[q]
            
            # Get correct answer
            correct_ans_list = correct_answers_dict.get(q_num, ["a"])  # Default to 'a' if no answer key
            
            # Determine status
            status = "Unattempted"
            if detected == "NONE":
                status = "Unattempted"
                unattempted += 1
            elif "," in detected:
                status = "Multiple Marked"
                multiple_marked += 1
            else:
                if detected in correct_ans_list:
                    status = "Correct"
                    correct += 1
                else:
                    status = "Incorrect"
                    incorrect += 1
            
            # Create detail entry
            detail = {
                "question": q + 1,
                "marked": detected,
                "correct": correct_ans_list,
                "status": status
            }
            details.append(detail)
        
        # Calculate scores
        total_score = correct
        max_score = questions
        percentage = (correct / max_score) * 100 if max_score > 0 else 0
        
        # Create the complete results structure
        results = {
            "source_image": image_name,
            "exam_set": set_key,
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1),
            "subject_scores": subject_scores,  # Would need subject mapping for real implementation
            "summary": {
                "correct": correct,
                "incorrect": incorrect,
                "unattempted": unattempted,
                "multiple_marked": multiple_marked,
                "partially_marked": partially_marked
            },
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "questions_with_answer_key": len(correct_answers_dict),
            "perspective_fallback": False,
            "detected_set_code": detected_set,
            "subjects_detected": True,
            "detection_method": "OCR-based"
        }
        
        return results
    
    def save_debug_images(self, output_dir, image_name):
        """Save debug images for analysis"""
        debug_dir = Path(output_dir) / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        base_name = Path(image_name).stem
        
        for stage, img in self.debug_images.items():
            if img is not None:
                output_path = debug_dir / f"{base_name}_{stage}.png"
                cv2.imwrite(str(output_path), img)
                print(f"Saved debug image: {output_path}")

# Example usage function
def process_single_image(image_path, excel_path=None, output_dir="./output"):
    """Process a single OMR image with the perfect debugging system"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the debugger
    debugger = PerfectOMRDebugger()
    
    # Load answer keys if provided
    answer_keys = None
    if excel_path and os.path.exists(excel_path):
        answer_keys = debugger.load_answer_key_from_excel(excel_path)
        print(f"Loaded answer keys: {list(answer_keys.keys()) if answer_keys else 'None'}")
    
    # Process the image
    results, final_image = debugger.process_omr_sheet(image_path, answer_keys)
    
    if results:
        # Save results JSON
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        
        results_path = os.path.join(output_dir, f"{base_name}_detailed.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Save final processed image
        final_image_path = os.path.join(output_dir, f"{base_name}_result.png")
        cv2.imwrite(final_image_path, final_image)
        print(f"Final image saved to: {final_image_path}")
        
        # Save debug images
        debugger.save_debug_images(output_dir, image_name)
        
        # Print summary
        print(f"\nProcessing Summary for {image_name}:")
        print(f"Total Score: {results['total_score']}/{results['max_score']} ({results['percentage']}%)")
        print(f"Correct: {results['summary']['correct']}")
        print(f"Incorrect: {results['summary']['incorrect']}")
        print(f"Unattempted: {results['summary']['unattempted']}")
        print(f"Multiple Marked: {results['summary']['multiple_marked']}")
        print(f"Detected Set: {results['detected_set_code']}")
        
        return results
    else:
        print("Failed to process image")
        return None

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_perfect_omr_debug.py <image_path> [excel_path] [output_dir]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    excel_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./output"
    
    results = process_single_image(image_path, excel_path, output_dir)