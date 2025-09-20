"""
Perfect OMR Debug System
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

# Try to import pytesseract for OCR functionality
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("pytesseract not available. OCR functionality will be limited.")

class PerfectOMRDebugger:
    """Perfect OMR Debug System with advanced detection capabilities"""
    
    def __init__(self):
        self.debug_images = {}
    
    def detect_filled_bubbles(self, image):
        """Detect ONLY student-filled circles in answer areas (A,B,C,D zones)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Step 1: Find very dark areas (student pen/pencil marks)
        _, dark_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)  # Very dark threshold
        
        # Step 2: Focus only on circular regions in answer areas
        # Create a mask for expected answer bubble locations
        h, w = gray.shape
        answer_zones_mask = np.zeros_like(gray)
        
        # Define answer zone regions (where A,B,C,D bubbles should be)
        # Based on typical OMR sheet layout - 5 columns of questions
        for col in range(5):  # 5 columns of questions
            for row in range(20):  # 20 questions per column
                # Calculate question position
                col_x_start = int(w * (0.1 + col * 0.18))  # Column start
                col_x_end = int(w * (0.25 + col * 0.18))   # Column end
                row_y = int(h * (0.15 + row * 0.035))      # Row position
                
                # Mark the 4 answer bubble areas (A, B, C, D) for this question
                for choice in range(4):
                    bubble_x = col_x_start + int((col_x_end - col_x_start) * (0.3 + choice * 0.15))
                    bubble_y = row_y
                    
                    # Create circular mask for this answer bubble location
                    cv2.circle(answer_zones_mask, (bubble_x, bubble_y), 25, 255, -1)
        
        # Step 3: Only look for dark marks in answer zones
        answer_dark_marks = cv2.bitwise_and(dark_mask, answer_zones_mask)
        
        # Step 4: Clean up the marks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        answer_dark_marks = cv2.morphologyEx(answer_dark_marks, cv2.MORPH_CLOSE, kernel)
        answer_dark_marks = cv2.morphologyEx(answer_dark_marks, cv2.MORPH_OPEN, kernel)
        
        # Step 5: Find contours of student marks
        contours, _ = cv2.findContours(answer_dark_marks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_bubbles = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter by size - only actual bubble marks (50-600 pixels)
            if 50 <= area <= 600:
                # Check if it's roughly circular
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Must be reasonably circular (student bubble)
                    if circularity > 0.35:
                        # Get center and radius
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x), int(y))
                        
                        # Check if this is actually a dark student mark
                        mask_roi = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.circle(mask_roi, center, int(radius), 255, -1)
                        
                        # Calculate darkness in this bubble
                        roi_gray = cv2.bitwise_and(gray, gray, mask=mask_roi)
                        roi_pixels = roi_gray[roi_gray > 0]
                        
                        if len(roi_pixels) > 0:
                            avg_darkness = np.mean(roi_pixels)
                            
                            # Only keep very dark marks (student filling)
                            if avg_darkness < 120:  # Must be quite dark
                                # Calculate fill density
                                fill_pixels = cv2.countNonZero(cv2.bitwise_and(answer_dark_marks, mask_roi))
                                total_pixels = np.pi * radius * radius
                                fill_ratio = fill_pixels / total_pixels if total_pixels > 0 else 0
                                
                                # Must be well-filled to be a student answer
                                if fill_ratio > 0.3:  # At least 30% filled
                                    filled_bubbles.append({
                                        'center': center,
                                        'radius': int(radius),
                                        'fill_ratio': fill_ratio,
                                        'area': area,
                                        'circularity': circularity,
                                        'avg_darkness': avg_darkness
                                    })
        
        # Sort by darkness (darkest/best marks first)
        filled_bubbles.sort(key=lambda b: b['avg_darkness'])
        
        return filled_bubbles

def perfect_ocr_analysis(image):
    """Master OCR analysis with perspective correction and multiple strategies"""
    
    # Step 1: Perspective correction
    corrected_image = apply_perspective_correction(image)
    if corrected_image is None:
        corrected_image = image
        print("Using original image (perspective correction failed)")
    else:
        print("Applied perspective correction")
    
    # Step 2: Multiple OCR preprocessing strategies
    gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY) if len(corrected_image.shape) == 3 else corrected_image
    
    preprocessing_methods = {
        'original': gray,
        'enhanced': None,
        'binary': None,
        'otsu': None,
        'adaptive': None
    }
    
    # Enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    preprocessing_methods['enhanced'] = clahe.apply(gray)
    
    # Binary threshold
    _, preprocessing_methods['binary'] = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # OTSU threshold  
    _, preprocessing_methods['otsu'] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive threshold
    preprocessing_methods['adaptive'] = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Step 3: OCR with multiple configurations
    ocr_configs = [
        ('--psm 6 --oem 3', 'Page segmentation'),
        # ('--psm 11 --oem 3', 'Sparse text'),
        ('--psm 8 --oem 3 -c tessedit_char_whitelist=ABCD0123456789', 'Whitelist only'),
    ]
    
    all_questions = {}
    all_options = []
    
    for method_name, processed_img in preprocessing_methods.items():
        if processed_img is None:
            continue
            
        for config, config_name in ocr_configs:
            try:
                ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT, config=config)
                
                for i, text in enumerate(ocr_data['text']):
                    if not text or not text.strip():
                        continue
                        
                    text_clean = text.strip()
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    confidence = ocr_data['conf'][i]
                    
                    # Filter low confidence detections
                    if confidence < 50:
                        continue
                    
                    # Question numbers (1-100)
                    if text_clean.isdigit():
                        q_num = int(text_clean)
                        if 1 <= q_num <= 100:
                            if q_num not in all_questions or confidence > all_questions[q_num].get('confidence', 0):
                                all_questions[q_num] = {
                                    'x': x, 'y': y, 'w': w, 'h': h,
                                    'confidence': confidence,
                                    'method': method_name,
                                    'config': config_name
                                }
                    
                    # Option letters (A, B, C, D)
                    elif len(text_clean) == 1 and text_clean.upper() in ['A', 'B', 'C', 'D']:
                        all_options.append({
                            'letter': text_clean.upper(),
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'confidence': confidence,
                            'method': method_name,
                            'config': config_name
                        })
            except Exception as e:
                print(f"OCR failed for {method_name} + {config_name}: {e}")
    
    return corrected_image, all_questions, all_options

def apply_perspective_correction(image):
    """Apply perspective correction to straighten the OMR sheet"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular contour (likely the OMR sheet)
        largest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    largest_contour = approx
        
        if largest_contour is not None and max_area > image.shape[0] * image.shape[1] * 0.1:
            # Order the points: top-left, top-right, bottom-right, bottom-left
            pts = largest_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            # Top-left point has the smallest sum, bottom-right has the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            # Top-right point has the smallest difference, bottom-left has the largest difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            # Calculate the destination points
            width = 1200  # Standard width
            height = 1600  # Standard height
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            # Apply perspective transformation
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            return warped
    
    except Exception as e:
        print(f"Perspective correction failed: {e}")
    
    return None

def debug_perfect_omr_detection():
    """Perfect OMR detection with visual debugging"""
    if not OCR_AVAILABLE:
        print("Tesseract not available")
        return
    
    # Load test image
    image_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Set_A\Img1.jpeg"
    
    try:
        print("=== PERFECT OMR DETECTION SYSTEM ===")
        print(f"Loading image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        print(f"Original image size: {image.shape}")
        
        # Step 1: Perfect OCR Analysis with perspective correction
        print("\n=== STEP 1: PERFECT OCR ANALYSIS ===")
        corrected_image, questions, options = perfect_ocr_analysis(image)
        
        print(f"✅ Detected {len(questions)} questions")
        print(f"✅ Detected {len(options)} option letters")
        
        # Step 2: Advanced bubble detection
        print("\n=== STEP 2: ADVANCED BUBBLE DETECTION ===")
        debugger = PerfectOMRDebugger()
        filled_bubbles = debugger.detect_filled_bubbles(corrected_image)
        print(f"✅ Detected {len(filled_bubbles)} filled bubbles")
        
        # Step 3: Create comprehensive visualization
        print("\n=== STEP 3: CREATING VISUAL DEBUG ===")
        debug_image = corrected_image.copy()
        
        # First, draw answer zone guides (light gray circles)
        h, w = debug_image.shape[:2]
        for col in range(5):  # 5 columns
            for row in range(20):  # 20 questions per column
                col_x_start = int(w * (0.1 + col * 0.18))
                col_x_end = int(w * (0.25 + col * 0.18))
                row_y = int(h * (0.15 + row * 0.035))
                
                # Draw the 4 expected answer bubble positions (A,B,C,D)
                for choice in range(4):
                    bubble_x = col_x_start + int((col_x_end - col_x_start) * (0.3 + choice * 0.15))
                    bubble_y = row_y
                    
                    # Draw light guide circle
                    cv2.circle(debug_image, (bubble_x, bubble_y), 25, (200, 200, 200), 1)
        
        # Draw ONLY student-filled bubbles with THICK RED circles
        bubble_data = []
        for i, bubble in enumerate(filled_bubbles):
            center = bubble['center']
            radius = bubble['radius']
            fill_ratio = bubble['fill_ratio']
            avg_darkness = bubble['avg_darkness']
            
            # Draw THICK RED circle for student mark
            thickness = 5 if fill_ratio > 0.5 else 4
            cv2.circle(debug_image, center, radius + 10, (0, 0, 255), thickness)  # RED outer circle
            
            # Draw filled center
            cv2.circle(debug_image, center, 4, (0, 0, 255), -1)  # RED center dot
            
            # Add detailed info
            label = f"{i+1}: {fill_ratio:.2f} d:{avg_darkness:.0f}"
            cv2.putText(debug_image, label, (center[0]+20, center[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            bubble_data.append({
                'bubble_id': i+1,
                'center': center,
                'radius': radius,
                'fill_ratio': bubble['fill_ratio'],
                'area': bubble['area'],
                'circularity': bubble['circularity'],
                'avg_darkness': bubble['avg_darkness']
            })
        
        # Draw detected questions with GREEN rectangles
        for q_num, q_data in questions.items():
            x, y, w, h = q_data['x'], q_data['y'], q_data['w'], q_data['h']
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_image, f"Q{q_num}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detected options with BLUE rectangles  
        for opt in options:
            x, y, w, h = opt['x'], opt['y'], opt['w'], opt['h']
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_image, opt['letter'], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Step 4: Smart bubble-option association
        print("\n=== STEP 4: SMART BUBBLE-OPTION ASSOCIATION ===")
        associations = []
        
        for q_num, q_data in questions.items():
            q_x, q_y = q_data['x'], q_data['y']
            
            # Find options near this question
            nearby_options = []
            for opt in options:
                # More flexible spatial matching
                horizontal_ok = opt['x'] > q_x and opt['x'] - q_x < 800
                vertical_distance = abs(opt['y'] - q_y)
                
                # Flexible vertical matching based on question position
                if q_y < 100:  # Top questions
                    vertical_ok = vertical_distance < 400
                elif q_y > corrected_image.shape[0] - 200:  # Bottom questions
                    vertical_ok = vertical_distance < 300
                else:  # Middle questions
                    vertical_ok = vertical_distance < 150
                
                if horizontal_ok and vertical_ok:
                    nearby_options.append(opt)
            
            # Sort options by X coordinate
            nearby_options.sort(key=lambda o: o['x'])
            
            # Associate bubbles with each option
            for opt in nearby_options:
                opt_center_x = opt['x'] + opt['w'] // 2
                opt_center_y = opt['y'] + opt['h'] // 2
                
                # Find nearest bubble to this option
                best_bubble = None
                min_distance = float('inf')
                
                for bubble_idx, bubble in enumerate(filled_bubbles):
                    bubble_x, bubble_y = bubble['center']
                    
                    # Calculate distance from option to bubble
                    dx = bubble_x - opt_center_x
                    dy = bubble_y - opt_center_y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Look for bubbles below and to the right of option letter
                    if dy > -20 and dx > -30 and distance < 100 and distance < min_distance:
                        min_distance = distance
                        best_bubble = bubble_idx
                
                if best_bubble is not None:
                    association = {
                        'question': q_num,
                        'option': opt['letter'],
                        'bubble_id': best_bubble + 1,
                        'bubble_center': filled_bubbles[best_bubble]['center'],
                        'distance': min_distance,
                        'fill_ratio': filled_bubbles[best_bubble]['fill_ratio']
                    }
                    associations.append(association)
                    
                    # Draw connection line from option to bubble
                    opt_center = (opt_center_x, opt_center_y)
                    bubble_center = filled_bubbles[best_bubble]['center']
                    cv2.line(debug_image, opt_center, bubble_center, (255, 255, 0), 2)  # Yellow line
        
        # Step 5: Generate final answers
        print("\n=== STEP 5: FINAL ANSWERS ===")
        final_answers = {}
        
        for q_num in range(1, 101):
            question_associations = [a for a in associations if a['question'] == q_num]
            
            if len(question_associations) == 0:
                final_answers[q_num] = "NONE"
            elif len(question_associations) == 1:
                final_answers[q_num] = question_associations[0]['option']
            else:
                # Multiple bubbles filled - decide based on fill ratio
                best_association = max(question_associations, key=lambda a: a['fill_ratio'])
                final_answers[q_num] = best_association['option']
        
        # Step 6: Save results and visualizations
        print("\n=== STEP 6: SAVING RESULTS ===")
        
        # Save debug visualization
        cv2.imwrite("perfect_omr_debug.png", debug_image)
        print("✅ Saved visual debug: perfect_omr_debug.png")
        
        # Save detection data
        detection_data = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'total_bubbles_detected': len(filled_bubbles),
            'bubble_details': bubble_data,
            'questions_detected': len(questions),
            'question_details': {str(k): v for k, v in questions.items()},
            'options_detected': len(options),
            'option_details': options,
            'associations': associations,
            'final_answers': final_answers
        }
        
        with open('perfect_omr_results.json', 'w') as f:
            json.dump(detection_data, f, indent=2, default=str)
        print("✅ Saved detection data: perfect_omr_results.json")
        
        # Step 7: Print summary
        print("\n" + "="*50)
        print("PERFECT OMR DETECTION SUMMARY")
        print("="*50)
        
        print(f"📊 Student marks in answer zones: {len(filled_bubbles)}")
        print(f"📝 Questions detected: {len(questions)}")
        print(f"🔤 Options detected: {len(options)}")
        print(f"🔗 Associations made: {len(associations)}")
        
        # Show quality metrics for detected student marks
        if filled_bubbles:
            very_dark_marks = [b for b in filled_bubbles if b['avg_darkness'] < 80]
            well_filled_marks = [b for b in filled_bubbles if b['fill_ratio'] > 0.5]
            avg_fill_ratio = np.mean([b['fill_ratio'] for b in filled_bubbles])
            avg_darkness = np.mean([b['avg_darkness'] for b in filled_bubbles])
            
            print(f"🌚 Very dark marks: {len(very_dark_marks)}/{len(filled_bubbles)}")
            print(f"🎯 Well-filled marks: {len(well_filled_marks)}/{len(filled_bubbles)}")
            print(f"📈 Average fill ratio: {avg_fill_ratio:.3f}")
            print(f"🖤 Average darkness: {avg_darkness:.1f}")
        
        # Show some specific results
        print(f"\n📋 Sample Results:")
        for q in [1, 3, 5, 10]:
            if q in final_answers:
                status = "✅" if final_answers[q] != "NONE" else "❌"
                print(f"  Q{q}: {final_answers[q]} {status}")
        
        # Show answer distribution
        answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'NONE': 0}
        for ans in final_answers.values():
            if ans in answer_counts:
                answer_counts[ans] += 1
        
        print(f"\n📈 Answer Distribution:")
        for choice, count in answer_counts.items():
            percentage = (count / 100) * 100
            print(f"  {choice}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 Success Rate: {((100 - answer_counts['NONE']) / 100) * 100:.1f}%")
        print("="*50)
        
    except Exception as e:
        print(f"Perfect OMR detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_perfect_omr_detection()