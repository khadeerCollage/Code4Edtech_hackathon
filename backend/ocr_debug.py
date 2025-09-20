import cv2
import numpy as np
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    
from main2 import OMRProcessor

def debug_ocr_detection():
    """Debug OCR text detection step by step"""
    if not OCR_AVAILABLE:
        print("Tesseract not available")
        return
    
    # Load test image
    image_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Set_A\Img1.jpeg"
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        print(f"Loaded image: {image.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Test different preprocessing methods
        print("\n=== Testing different OCR preprocessing methods ===")
        
        methods = {
            'original': gray,
            'enhanced_contrast': None,
            'threshold_binary': None,
            'threshold_otsu': None,
            'adaptive_thresh': None,
            'morph_open': None
        }
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        methods['enhanced_contrast'] = clahe.apply(gray)
        
        # Binary threshold
        _, methods['threshold_binary'] = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # OTSU threshold
        _, methods['threshold_otsu'] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive threshold
        methods['adaptive_thresh'] = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological opening
        kernel = np.ones((2,2), np.uint8)
        methods['morph_open'] = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Test each method with different OCR configs
        configs = [
            ('--psm 6 --oem 3', 'Auto page segmentation'),
            ('--psm 8 --oem 3', 'Single word'),
            ('--psm 7 --oem 3', 'Single text line'),
            ('--psm 11 --oem 3', 'Sparse text'),
            ('--psm 12 --oem 3', 'Sparse text OSD'),
            ('--psm 6 --oem 3 -c tessedit_char_whitelist=ABCD0123456789', 'Whitelist chars'),
            ('--psm 8 --oem 3 -c tessedit_char_whitelist=ABCD', 'Only letters'),
            ('--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789', 'Only numbers'),
        ]
        
        best_results = {}
        
        for method_name, processed_img in methods.items():
            if processed_img is None:
                continue
                
            print(f"\n--- Testing method: {method_name} ---")
            
            for config, config_desc in configs:
                try:
                    # Get OCR data
                    ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT, config=config)
                    
                    # Count detections
                    questions = []
                    options = []
                    all_text = []
                    
                    for i, text in enumerate(ocr_data['text']):
                        if not text or not text.strip():
                            continue
                            
                        text_clean = text.strip()
                        all_text.append(text_clean)
                        
                        # Check for question numbers
                        if text_clean.isdigit():
                            q_num = int(text_clean)
                            if 1 <= q_num <= 100:
                                questions.append(q_num)
                        
                        # Check for option letters
                        elif len(text_clean) == 1 and text_clean.upper() in ['A', 'B', 'C', 'D']:
                            options.append(text_clean.upper())
                    
                    q_count = len(set(questions))
                    opt_count = len(options)
                    total_text = len(all_text)
                    
                    print(f"  {config_desc}: {q_count} questions, {opt_count} options, {total_text} total")
                    
                    # Store best results
                    key = f"{method_name}_{config_desc}"
                    best_results[key] = {
                        'questions': q_count,
                        'options': opt_count,
                        'total': total_text,
                        'question_list': sorted(set(questions))[:10],  # First 10
                        'option_list': options[:20],  # First 20
                        'all_text': all_text[:20]  # First 20
                    }
                    
                except Exception as e:
                    print(f"  {config_desc}: ERROR - {e}")
        
        # Show best results
        print("\n=== BEST RESULTS SUMMARY ===")
        sorted_results = sorted(best_results.items(), 
                              key=lambda x: (x[1]['questions'] + x[1]['options']), 
                              reverse=True)
        
        for key, result in sorted_results[:5]:  # Top 5
            print(f"\n{key}:")
            print(f"  Questions: {result['questions']} - {result['question_list']}")
            print(f"  Options: {result['options']} - {result['option_list']}")
            print(f"  Sample text: {result['all_text'][:10]}")
        
        # Test specific image regions
        print("\n=== TESTING SPECIFIC REGIONS ===")
        h, w = gray.shape
        
        # Focus on top-left area (should have Q1-Q10)
        region = gray[int(h*0.1):int(h*0.5), int(w*0.1):int(w*0.4)]
        cv2.imwrite('debug_region.png', region)
        
        config = '--psm 6 --oem 3'
        try:
            region_data = pytesseract.image_to_data(region, output_type=pytesseract.Output.DICT, config=config)
            region_text = [t.strip() for t in region_data['text'] if t and t.strip()]
            print(f"Top-left region text: {region_text[:20]}")
        except Exception as e:
            print(f"Region OCR failed: {e}")
            
    except Exception as e:
        print(f"Debug failed: {e}")

if __name__ == "__main__":
    debug_ocr_detection()