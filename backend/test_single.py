#!/usr/bin/env python3

import cv2
import numpy as np
from main2 import OMRProcessor
import os

def test_single_image():
    """Test single image with OCR-based detection"""
    
    # Load test image
    image_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Set_A\Img1.jpeg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print("Testing OMR processing with OCR-based detection...")
    print(f"Processing: {image_path}")
    
    # Initialize processor
    processor = OMRProcessor()
    
    # Process the image
    try:
        results = processor.process_image(image_path)
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Show set detection
        if 'set_code' in results:
            print(f"Detected Set: {results['set_code']}")
        
        # Show answers
        if 'answers' in results:
            answers = results['answers']
            
            # Count answer distribution
            answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'NONE': 0, 'MULTIPLE': 0}
            for ans in answers.values():
                if ans in answer_counts:
                    answer_counts[ans] += 1
                else:
                    answer_counts['NONE'] += 1
            
            print(f"\nAnswer Distribution:")
            for choice, count in answer_counts.items():
                print(f"  {choice}: {count}")
            
            # Show specific answers for first 10 questions
            print(f"\nFirst 10 Questions:")
            for i in range(1, 11):
                q_key = str(i)
                answer = answers.get(q_key, "NONE")
                print(f"  Q{i}: {answer}")
            
            # Show answers for specific questions that should be marked
            print(f"\nSpecific Test Cases:")
            print(f"  Q1: {answers.get('1', 'NONE')} (Expected: A)")
            print(f"  Q3: {answers.get('3', 'NONE')} (Expected: B)")
            
            # Show total processed
            total_answered = sum(1 for ans in answers.values() if ans not in ['NONE', 'MULTIPLE'])
            print(f"\nTotal Questions with Answers: {total_answered}/100")
            
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_image()