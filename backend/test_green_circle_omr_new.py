"""
Enhanced Green Circle OMR Processor with Improved Bubble Detection and Evaluation
Focus on accurate circle detection, classification and simple JSON reporting
Features:
 - Optimized bubble detection with adaptive thresholds
 - Better handling of marked circles, partial marks, and multiple marks
 - Cleaner, simplified JSON output format
 - Visual debugging of detection process
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

def detect_and_evaluate_omr(image_path, excel_path, output_dir):
    """
    Main function to detect bubbles and evaluate OMR sheet with Excel integration
    Returns comprehensive evaluation report
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Step 1: Detect set from image
    detected_set = detect_set_from_image(image_path)
    print(f"Detected Set: {detected_set}")
    
    # Step 2: Load answer key from Excel based on set
    answer_key, subject_info = read_excel_answer_key_by_set(excel_path, detected_set)
    
    # Step 3: Enhanced bubble detection
    circles_data = detect_bubbles_enhanced(image)
    print(f"Detected {len(circles_data)} total circles")
    
    # Step 4: Filter and group into questions  
    questions_data = group_circles_to_questions(circles_data, image.shape)
    print(f"Grouped into {len(questions_data)} questions")
    
    # Step 5: Evaluate against answer key with subject breakdown
    evaluation = evaluate_by_subjects(questions_data, answer_key, subject_info)
    
    # Step 6: Create comprehensive JSON report
    report = create_comprehensive_report(image_path, detected_set, evaluation, subject_info, output_dir)
    
    # Step 7: Save debug visualization
    save_debug_visualization(image, circles_data, questions_data, output_dir)
    
    return report

def detect_bubbles_enhanced(image):
    """
    Enhanced bubble detection with dual approach:
      - Try HoughCircles first (fast, clean images).
      - Fallback to contour-based detection if Hough fails or detects too few.
    Returns: list of bubble dicts with metrics (center, radius, fill_ratio, confidence, etc.)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Preprocessing: CLAHE + blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (3,3), 0)

    circles_data = []

    # ---------- Method 1: HoughCircles ----------
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=22,
        param1=60, param2=25, minRadius=8, maxRadius=25
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if y < 80:  # skip header/top margin
                continue
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r-2, 255, -1)
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            dark_pixels = cv2.countNonZero(cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
            fill_ratio = dark_pixels / (np.pi * (r**2) + 1e-6)

            circles_data.append({
                "center": (x, y),
                "radius": r,
                "fill_ratio": round(fill_ratio, 3),
                "is_marked": False,  # will be updated later
                "confidence": 0.0
            })

    # ---------- Method 2: Contour fallback ----------
    if len(circles_data) < 50:  # too few circles, fallback
        print("⚠️ Hough weak, using contour fallback...")
        thr = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50 or area > 3000:  # filter noise
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            x, y, r = int(x), int(y), int(r)
            if r < 7 or r > 25:  # size sanity check
                continue
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r-1, 255, -1)
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            dark_pixels = cv2.countNonZero(cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
            fill_ratio = dark_pixels / (np.pi * (r**2) + 1e-6)

            circles_data.append({
                "center": (x, y),
                "radius": r,
                "fill_ratio": round(fill_ratio, 3),
                "is_marked": False,
                "confidence": 0.0
            })

    # ---------- Dynamic thresholding ----------
    if not circles_data:
        return []

    fill_vals = np.array([c['fill_ratio'] for c in circles_data])
    if len(fill_vals) > 5:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=42).fit(fill_vals.reshape(-1,1))
        centers = sorted(km.cluster_centers_.flatten())
        thr = (centers[0] + centers[1]) / 2
    else:
        thr = np.mean(fill_vals)

    for c in circles_data:
        c["is_marked"] = c["fill_ratio"] >= thr
        c["confidence"] = round(abs(c["fill_ratio"] - thr) / max(thr, 1e-6), 2)

    print(f"Dynamic threshold={thr:.3f}, Detected={len(circles_data)} bubbles")
    return circles_data

def detect_set_from_image(image_path):
    """Detect whether the image is Set A or Set B"""
    # Extract set information from image name or path
    image_name = os.path.basename(image_path).upper()
    
    if "SET_A" in image_name or "SETA" in image_name:
        return "A"
    elif "SET_B" in image_name or "SETB" in image_name:
        return "B"
    
    # Check from folder path
    if "SET_A" in image_path.upper() or "SETA" in image_path.upper():
        return "A"
    elif "SET_B" in image_path.upper() or "SETB" in image_path.upper():
        return "B"
    
    # Default to A if cannot determine
    print("Warning: Could not determine set from image path, defaulting to Set A")
    return "A"

def read_excel_answer_key_by_set(excel_path, detected_set):
    """Read answer key from Excel file based on detected set"""
    try:
        # Read all sheets from Excel file
        excel_sheets = pd.read_excel(excel_path, sheet_name=None)
        
        print(f"Available sheets: {list(excel_sheets.keys())}")
        print(f"Detected Set: {detected_set}")
        
        # Look for the appropriate sheet
        target_sheet = None
        for sheet_name in excel_sheets.keys():
            if f"Set {detected_set}" in sheet_name or f"SET {detected_set}" in sheet_name:
                target_sheet = sheet_name
                break
            elif detected_set in sheet_name.upper():
                target_sheet = sheet_name
                break
        
        # If no specific sheet found, use the first sheet
        if target_sheet is None:
            target_sheet = list(excel_sheets.keys())[0]
            print(f"Using default sheet: {target_sheet}")
        else:
            print(f"Using sheet: {target_sheet}")
        
        df = excel_sheets[target_sheet]
        print(f"Sheet shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Parse answer key by subjects
        answer_key = {}
        subject_info = {}
        
        # Process each column (subject)
        for col in df.columns:
            if col.strip() and not col.startswith('Unnamed'):
                subject_name = col.strip()
                print(f"Processing subject: {subject_name}")
                
                subject_answers = []
                for index, row in df.iterrows():
                    cell_value = str(row[col]).strip()
                    if '-' in cell_value and not pd.isna(row[col]):
                        # Parse format like "1 - a" or "21 - c"
                        parts = cell_value.split('-')
                        if len(parts) >= 2:
                            question_num = parts[0].strip()
                            # Remove any non-numeric characters from question number
                            question_num = ''.join(filter(str.isdigit, question_num))
                            answer = parts[1].strip().lower()
                            
                            if question_num and answer in ['a', 'b', 'c', 'd']:
                                answer_key[question_num] = [answer]
                                subject_answers.append(int(question_num))
                
                if subject_answers:
                    subject_info[subject_name] = {
                        'questions': sorted(subject_answers),
                        'range': f"{min(subject_answers)}-{max(subject_answers)}",
                        'count': len(subject_answers)
                    }
        
        print(f"\n=== SUBJECT BREAKDOWN ===")
        for subject, info in subject_info.items():
            print(f"{subject}: Questions {info['range']} ({info['count']} questions)")
        
        print(f"\nTotal answers loaded: {len(answer_key)} questions")
        return answer_key, subject_info
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}

def evaluate_by_subjects(questions_data, answer_key, subject_info):
    """Evaluate answers by subject sections"""
    results = []
    summary = Counter()
    subject_results = {}
    
    # Initialize subject results
    for subject, info in subject_info.items():
        subject_results[subject] = {
            'total': info['count'],
            'correct': 0,
            'incorrect': 0,
            'unattempted': 0,
            'multiple_marked': 0,
            'details': []
        }
    
    # Evaluate each question
    for q_num, data in questions_data.items():
        marked = data['marked']
        correct = answer_key.get(str(q_num), [])
        
        # FIXED: Correct logic order for status determination
        if not marked:
            status = "unattempted"
            summary['unattempted'] += 1
        elif set(marked) == set(correct):
            status = "correct"  
            summary['correct'] += 1
        elif len(marked) > 1:
            # Only mark as multiple if it's NOT correct
            status = "multiple_marked"
            summary['multiple_marked'] += 1
        else:
            status = "incorrect"
            summary['incorrect'] += 1
        
        # Find which subject this question belongs to
        question_subject = "Unknown"
        for subject, info in subject_info.items():
            if q_num in info['questions']:
                question_subject = subject
                subject_results[subject][status] += 1
                subject_results[subject]['details'].append({
                    'question': q_num,
                    'marked': marked,
                    'correct': correct,
                    'status': status
                })
                break
        
        results.append({
            "question": q_num,
            "subject": question_subject,
            "marked": ",".join(marked) if marked else "NONE",
            "correct": ",".join(correct),
            "status": status
        })
    
    return {
        'results': results, 
        'summary': dict(summary),
        'subject_results': subject_results
    }

def group_circles_to_questions(circles_data, image_shape, target_questions=100):
    """
    Group detected bubbles into questions using clustering.
    Each row (question) → 4 options (A-D).
    If multiple are marked, keep only the bubble with the highest fill_ratio.
    If none pass the global threshold, pick the max anyway (fallback).
    """
    if not circles_data:
        return {}

    centers = np.array([c['center'] for c in circles_data])
    ys = centers[:, 1].reshape(-1, 1)

    from sklearn.cluster import AgglomerativeClustering
    row_clust = AgglomerativeClustering(
        n_clusters=min(target_questions, len(circles_data)//4),
        linkage="ward"
    ).fit(ys)

    row_groups = {}
    for idx, label in enumerate(row_clust.labels_):
        row_groups.setdefault(label, []).append(circles_data[idx])

    questions = {}
    q_num = 1
    for row_id, row_circles in sorted(row_groups.items(),
                                      key=lambda x: np.mean([c['center'][1] for c in x[1]])):
        row_sorted = sorted(row_circles, key=lambda c: c['center'][0])
        if len(row_sorted) < 4:
            continue

        for i in range(0, len(row_sorted)-3, 4):
            group = row_sorted[i:i+4]

            # Mark candidates based on global threshold
            marked_candidates = [
                (["a", "b", "c", "d"][j], c["fill_ratio"])
                for j, c in enumerate(group) if c["is_marked"]
            ]

            if marked_candidates:
                # normal case: pick the darkest
                best_option = max(marked_candidates, key=lambda x: x[1])[0]
                marked = [best_option]
            else:
                # fallback: pick the darkest anyway (relative marking)
                best_circle = max(group, key=lambda c: c["fill_ratio"])
                best_idx = group.index(best_circle)
                best_option = ["a", "b", "c", "d"][best_idx]

                # only keep if it's reasonably darker than others
                fills = [c["fill_ratio"] for c in group]
                if best_circle["fill_ratio"] > np.mean(fills) * 1.2:
                    marked = [best_option]
                else:
                    marked = []  # truly blank

            questions[q_num] = {
                "marked": marked,
                "circles": group
            }
            q_num += 1
            if q_num > target_questions:
                break
        if q_num > target_questions:
            break

    print(f"Grouped into {len(questions)} questions")
    return questions


def create_comprehensive_report(image_path, detected_set, evaluation, subject_info, output_dir):
    """Create comprehensive JSON evaluation report with subject breakdown"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Calculate subject percentages
    subject_percentages = {}
    for subject, results in evaluation['subject_results'].items():
        total = results['total']
        correct = results['correct']
        percentage = round((correct / total) * 100, 1) if total > 0 else 0
        subject_percentages[subject] = {
            'total_questions': total,
            'correct': correct,
            'percentage': percentage,
            'incorrect': results['incorrect'],
            'unattempted': results['unattempted'],
            'multiple_marked': results['multiple_marked']
        }
    
    # Comprehensive report format
    report = {
        "image": os.path.basename(image_path),
        "detected_set": detected_set,
        "timestamp": datetime.now().isoformat()[:19],
        "total_questions": len(evaluation['results']),
        "score": evaluation['summary'].get('correct', 0),
        "percentage": round((evaluation['summary'].get('correct', 0) / len(evaluation['results'])) * 100, 1) if evaluation['results'] else 0,
        "overall_summary": evaluation['summary'],
        "subject_breakdown": subject_percentages,
        "subject_details": evaluation['subject_results'],
        "detailed_results": evaluation['results']
    }
    
    # Save comprehensive report
    report_path = os.path.join(output_dir, f"{base_name}_comprehensive_evaluation.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Comprehensive report saved: {report_path}")
    return report

def create_evaluation_report(image_path, evaluation, output_dir):
    """Create simple JSON evaluation report"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Simple report format
    report = {
        "image": os.path.basename(image_path),
        "timestamp": datetime.now().isoformat()[:19],
        "total_questions": len(evaluation['results']),
        "score": evaluation['summary'].get('correct', 0),
        "percentage": round((evaluation['summary'].get('correct', 0) / len(evaluation['results'])) * 100, 1) if evaluation['results'] else 0,
        "summary": evaluation['summary'],
        "details": evaluation['results']
    }
    
    # Save report
    report_path = os.path.join(output_dir, f"{base_name}_simple_evaluation.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved: {report_path}")
    return report

def save_debug_visualization(image, circles_data, questions_data, output_dir):
    """Save debug visualization showing detected circles"""
    debug_img = image.copy()
    
    for circle in circles_data:
        center = circle['center']
        radius = circle['radius']
        
        # Color based on marking status
        if circle['is_marked']:
            color = (0, 0, 255)  # Red for marked
            thickness = 3
        else:
            color = (0, 255, 0)  # Green for unmarked
            thickness = 1
        
        cv2.circle(debug_img, center, radius, color, thickness)
        
        # Add fill ratio text for marked circles
        if circle['is_marked']:
            cv2.putText(debug_img, f"{circle['fill_ratio']:.2f}", 
                       (center[0] - 15, center[1] - radius - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Save debug image
    base_name = os.path.splitext(os.path.basename(debug_img))[0] if hasattr(debug_img, '__name__') else "debug"
    debug_path = os.path.join(output_dir, f"debug_visualization.png")
    cv2.imwrite(debug_path, debug_img)
    print(f"Debug image saved: {debug_path}")

def main():
    """Main function - Excel-based OMR evaluation with subject breakdown"""
    
    # Configuration
    image_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Set_A\Img4.jpeg"
    excel_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Key (Set A and B).xlsx"
    output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
    
    print("=== EXCEL-BASED OMR EVALUATION WITH SUBJECT BREAKDOWN ===")
    
    try:
        # Run comprehensive evaluation
        report = detect_and_evaluate_omr(image_path, excel_path, output_dir)
        
        # Print comprehensive results
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Image: {report['image']}")
        print(f"Detected Set: {report['detected_set']}")
        print(f"Total Questions: {report['total_questions']}")
        print(f"Overall Score: {report['score']} ({report['percentage']}%)")
        
        print(f"\n=== SUBJECT BREAKDOWN ===")
        for subject, results in report['subject_breakdown'].items():
            print(f"{subject}:")
            print(f"  Score: {results['correct']}/{results['total_questions']} ({results['percentage']}%)")
            print(f"  Incorrect: {results['incorrect']}, Unattempted: {results['unattempted']}")
            if results['multiple_marked'] > 0:
                print(f"  Multiple marked: {results['multiple_marked']}")
        
        print(f"\n=== OVERALL SUMMARY ===")
        print(f"Correct: {report['overall_summary'].get('correct', 0)}")
        print(f"Incorrect: {report['overall_summary'].get('incorrect', 0)}")
        print(f"Unattempted: {report['overall_summary'].get('unattempted', 0)}")
        print(f"Multiple marked: {report['overall_summary'].get('multiple_marked', 0)}")
        
        # Show first few detailed results
        print(f"\n=== SAMPLE RESULTS ===")
        for detail in report['detailed_results'][:10]:
            q = detail['question']
            subj = detail['subject']
            marked = detail['marked']
            correct = detail['correct']
            status = detail['status']
            print(f"Q{q} ({subj}): {marked} -> {correct} ({status})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()