"""
Green Circle OMR Processing Test
Specifically designed for the green circle OMR format with black filled marks
"""

import os
import cv2
import json
import numpy as np
from enhanced_perfect_omr_debug import PerfectOMRDebugger

def test_green_circle_omr(image_path):
    """Test the green circle OMR processing with the provided image"""
    
    print("=" * 70)
    print("GREEN CIRCLE OMR PROCESSING TEST")
    print("=" * 70)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    # Create output directory
    output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize the debugger
    debugger = PerfectOMRDebugger()
    
    # Create sample answer key for testing
    sample_answer_key = {
        'Set_A': {
            '1': ['a'], '2': ['b'], '3': ['c'], '4': ['d'], '5': ['a'],
            '6': ['b'], '7': ['c'], '8': ['d'], '9': ['a'], '10': ['b'],
            '11': ['c'], '12': ['d'], '13': ['a'], '14': ['b'], '15': ['c'],
            '16': ['d'], '17': ['a'], '18': ['b'], '19': ['c'], '20': ['d']
        }
    }
    
    try:
        # Process the image
        print("\nStep 1: Reading and analyzing image...")
        results, final_image = debugger.process_omr_sheet(image_path, sample_answer_key, questions=20, choices=4)
        
        if results:
            # Save results
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            
            # Save JSON results
            results_path = os.path.join(output_dir, f"{base_name}_green_circle_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save processed image
            final_image_path = os.path.join(output_dir, f"{base_name}_processed.png")
            cv2.imwrite(final_image_path, final_image)
            
            # Save debug images
            debugger.save_debug_images(output_dir, image_name)
            
            print(f"\n✅ PROCESSING SUCCESSFUL!")
            print(f"Results saved to: {results_path}")
            print(f"Processed image saved to: {final_image_path}")
            
            # Display results summary
            print(f"\n📊 RESULTS SUMMARY:")
            print(f"Source Image: {results['source_image']}")
            print(f"Detected Set: {results['detected_set_code']}")
            print(f"Total Score: {results['total_score']}/{results['max_score']}")
            print(f"Percentage: {results['percentage']}%")
            
            summary = results['summary']
            print(f"\nAnswer Breakdown:")
            print(f"✅ Correct: {summary['correct']}")
            print(f"❌ Incorrect: {summary['incorrect']}")
            print(f"⭕ Unattempted: {summary['unattempted']}")
            print(f"🔄 Multiple Marked: {summary['multiple_marked']}")
            
            # Show first 20 detailed results
            print(f"\n📝 Detailed Question Analysis:")
            print("Q#  | Marked | Correct | Status")
            print("-" * 35)
            
            for detail in results['details'][:20]:
                q_num = f"Q{detail['question']:2d}"
                marked = detail['marked'][:6] if detail['marked'] != 'NONE' else 'NONE  '
                correct = ','.join(detail['correct'])[:6]
                status = detail['status'][:12]
                
                # Add status emoji
                status_emoji = {
                    'Correct': '✅',
                    'Incorrect': '❌',
                    'Unattempted': '⭕',
                    'Multiple Marked': '🔄'
                }.get(status, '❓')
                
                print(f"{q_num} | {marked:6} | {correct:6} | {status_emoji} {status}")
            
            # Calculate and show answer distribution
            answer_counts = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'NONE': 0, 'MULTI': 0}
            for detail in results['details']:
                marked = detail['marked']
                if marked == 'NONE':
                    answer_counts['NONE'] += 1
                elif ',' in marked:
                    answer_counts['MULTI'] += 1
                elif marked in answer_counts:
                    answer_counts[marked] += 1
            
            print(f"\n📈 Answer Distribution:")
            for choice, count in answer_counts.items():
                percentage = (count / len(results['details'])) * 100
                bar = '█' * int(percentage // 5)
                print(f"{choice:4}: {count:2d} ({percentage:5.1f}%) {bar}")
            
            return results
            
        else:
            print("❌ Processing failed - no results returned")
            return None
            
    except Exception as e:
        print(f"❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_detailed_analysis_image(image_path, output_dir):
    """Create a detailed analysis image showing the detection process"""
    
    print(f"\n🔍 Creating detailed analysis image...")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image for analysis")
        return
    
    debugger = PerfectOMRDebugger()
    
    # Detect green circles and marks
    green_circles, marked_circles = debugger.detect_green_circles_and_black_marks(img)
    
    # Create analysis image
    analysis_img = img.copy()
    
    # Draw all green circles in green
    for circle in green_circles:
        center = circle['center']
        radius = circle['radius']
        cv2.circle(analysis_img, center, radius, (0, 255, 0), 2)
        cv2.circle(analysis_img, center, 3, (0, 255, 0), -1)
    
    # Draw marked circles in red
    for i, circle in enumerate(marked_circles):
        center = circle['center']
        radius = circle['radius']
        fill_ratio = circle['fill_ratio']
        
        cv2.circle(analysis_img, center, radius, (0, 0, 255), 3)
        cv2.circle(analysis_img, center, 5, (0, 0, 255), -1)
        
        # Add fill ratio text
        cv2.putText(analysis_img, f"{fill_ratio:.2f}", 
                   (center[0] + radius + 5, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Add summary text
    cv2.putText(analysis_img, f"Green Circles: {len(green_circles)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(analysis_img, f"Marked Circles: {len(marked_circles)}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Save analysis image
    analysis_path = os.path.join(output_dir, "detailed_analysis.png")
    cv2.imwrite(analysis_path, analysis_img)
    print(f"Analysis image saved to: {analysis_path}")

def main():
    """Main function to test green circle OMR processing"""
    
    # Test with the provided image (save it first)
    test_image_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\test_green_circle_omr.png"
    
    # For now, we'll test with an existing image from the dataset
    fallback_image = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Set_A\Img6.jpeg"
    
    if os.path.exists(test_image_path):
        image_path = test_image_path
        print("Using provided green circle OMR image")
    elif os.path.exists(fallback_image):
        image_path = fallback_image
        print("Using fallback image from dataset")
    else:
        print("No test image available")
        return
    
    # Process the image
    results = test_green_circle_omr(image_path)
    
    if results:
        # Create detailed analysis
        output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
        create_detailed_analysis_image(image_path, output_dir)
        
        print(f"\n🎉 GREEN CIRCLE OMR PROCESSING COMPLETE!")
        print(f"📁 All files saved to: {output_dir}")
        print(f"\nThe system is now optimized for green circle OMR sheets!")
        print(f"It can detect:")
        print(f"  ✅ Green circular answer bubbles")
        print(f"  ✅ Black filled marks inside circles") 
        print(f"  ✅ Multiple marked answers")
        print(f"  ✅ Unattempted questions")
        print(f"  ✅ Answer comparison with answer key")

if __name__ == "__main__":
    main()