# # """
# # Green Circle OMR Processing Test
# # Specifically designed for the green circle OMR format with black filled marks
# # """

# # import os
# # import cv2
# # import json
# # import numpy as np
# # from enhanced_perfect_omr_debug import PerfectOMRDebugger

# # def test_green_circle_omr(image_path):
# #     """Test the green circle OMR processing with the provided image"""
    
# #     print("=" * 70)
# #     print("GREEN CIRCLE OMR PROCESSING TEST")
# #     print("=" * 70)
    
# #     if not os.path.exists(image_path):
# #         print(f"Error: Image not found: {image_path}")
# #         return
    
# #     # Create output directory
# #     output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     print(f"Processing: {image_path}")
# #     print(f"Output directory: {output_dir}")
    
# #     # Initialize the debugger
# #     debugger = PerfectOMRDebugger()
    
# #     # Create sample answer key for testing
# #     sample_answer_key = {
# #         'Set_A': {
# #             '1': ['a'], '2': ['c'], '3': ['c'], '4': ['c'], '5': ['c'],
# #             '6': ['a'], '7': ['c'], '8': ['c'], '9': ['b'], '10': ['c'],
# #             '11': ['a'], '12': ['a'], '13': ['d'], '14': ['a'], '15': ['a'],
# #             '16': ['b'], '17': ['a','b','c','d'], '18': ['d'], '19': ['a'], '20': ['b']
# #         }
# #     }
    
# #     try:
# #         # Process the image
# #         print("\nStep 1: Reading and analyzing image...")
# #         results, final_image = debugger.process_omr_sheet(image_path, sample_answer_key, questions=20, choices=4)
        
# #         if results:
# #             # Save results
# #             image_name = os.path.basename(image_path)
# #             base_name = os.path.splitext(image_name)[0]
            
# #             # Save JSON results
# #             results_path = os.path.join(output_dir, f"{base_name}_green_circle_results.json")
# #             with open(results_path, 'w') as f:
# #                 json.dump(results, f, indent=2)
            
# #             # Save processed image
# #             final_image_path = os.path.join(output_dir, f"{base_name}_processed.png")
# #             cv2.imwrite(final_image_path, final_image)
            
# #             # Save debug images
# #             debugger.save_debug_images(output_dir, image_name)
            
# #             print(f"\n✅ PROCESSING SUCCESSFUL!")
# #             print(f"Results saved to: {results_path}")
# #             print(f"Processed image saved to: {final_image_path}")
            
# #             # Display results summary
# #             print(f"\n📊 RESULTS SUMMARY:")
# #             print(f"Source Image: {results['source_image']}")
# #             print(f"Detected Set: {results['detected_set_code']}")
# #             print(f"Total Score: {results['total_score']}/{results['max_score']}")
# #             print(f"Percentage: {results['percentage']}%")
            
# #             summary = results['summary']
# #             print(f"\nAnswer Breakdown:")
# #             print(f"✅ Correct: {summary['correct']}")
# #             print(f"❌ Incorrect: {summary['incorrect']}")
# #             print(f"⭕ Unattempted: {summary['unattempted']}")
# #             print(f"🔄 Multiple Marked: {summary['multiple_marked']}")
            
# #             # Show first 20 detailed results
# #             print(f"\n📝 Detailed Question Analysis:")
# #             print("Q#  | Marked | Correct | Status")
# #             print("-" * 35)
            
# #             for detail in results['details'][:20]:
# #                 q_num = f"Q{detail['question']:2d}"
# #                 marked = detail['marked'][:6] if detail['marked'] != 'NONE' else 'NONE  '
# #                 correct = ','.join(detail['correct'])[:6]
# #                 status = detail['status'][:12]
                
# #                 # Add status emoji
# #                 status_emoji = {
# #                     'Correct': '✅',
# #                     'Incorrect': '❌',
# #                     'Unattempted': '⭕',
# #                     'Multiple Marked': '🔄'
# #                 }.get(status, '❓')
                
# #                 print(f"{q_num} | {marked:6} | {correct:6} | {status_emoji} {status}")
            
# #             # Calculate and show answer distribution
# #             answer_counts = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'NONE': 0, 'MULTI': 0}
# #             for detail in results['details']:
# #                 marked = detail['marked']
# #                 if marked == 'NONE':
# #                     answer_counts['NONE'] += 1
# #                 elif ',' in marked:
# #                     answer_counts['MULTI'] += 1
# #                 elif marked in answer_counts:
# #                     answer_counts[marked] += 1
            
# #             print(f"\n📈 Answer Distribution:")
# #             for choice, count in answer_counts.items():
# #                 percentage = (count / len(results['details'])) * 100
# #                 bar = '█' * int(percentage // 5)
# #                 print(f"{choice:4}: {count:2d} ({percentage:5.1f}%) {bar}")
            
# #             return results
            
# #         else:
# #             print("❌ Processing failed - no results returned")
# #             return None
            
# #     except Exception as e:
# #         print(f"❌ ERROR during processing: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         return None

# # def create_detailed_analysis_image(image_path, output_dir):
# #     """Create a detailed analysis image showing the detection process"""
    
# #     print(f"\n🔍 Creating detailed analysis image...")
    
# #     # Read the image
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         print("Could not read image for analysis")
# #         return
    
# #     debugger = PerfectOMRDebugger()
    
# #     # Detect green circles and marks
# #     green_circles, marked_circles = debugger.detect_green_circles_and_black_marks(img)
    
# #     # Create analysis image
# #     analysis_img = img.copy()
    
# #     # Draw all green circles in green
# #     for circle in green_circles:
# #         center = circle['center']
# #         radius = circle['radius']
# #         cv2.circle(analysis_img, center, radius, (0, 255, 0), 2)
# #         cv2.circle(analysis_img, center, 3, (0, 255, 0), -1)
    
# #     # Draw marked circles in red
# #     for i, circle in enumerate(marked_circles):
# #         center = circle['center']
# #         radius = circle['radius']
# #         fill_ratio = circle['fill_ratio']
        
# #         cv2.circle(analysis_img, center, radius, (0, 0, 255), 3)
# #         cv2.circle(analysis_img, center, 5, (0, 0, 255), -1)
        
# #         # Add fill ratio text
# #         cv2.putText(analysis_img, f"{fill_ratio:.2f}", 
# #                    (center[0] + radius + 5, center[1]), 
# #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
# #     # Add summary text
# #     cv2.putText(analysis_img, f"Green Circles: {len(green_circles)}", 
# #                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #     cv2.putText(analysis_img, f"Marked Circles: {len(marked_circles)}", 
# #                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
# #     # Save analysis image
# #     analysis_path = os.path.join(output_dir, "detailed_analysis.png")
# #     cv2.imwrite(analysis_path, analysis_img)
# #     print(f"Analysis image saved to: {analysis_path}")

# # def main():
# #     """Main function to test green circle OMR processing"""
    
# #     # Test with the provided image (save it first)
# #     test_image_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\test_green_circle_omr.png"
    
# #     # For now, we'll test with an existing image from the dataset
# #     fallback_image = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Set_A\Img6.jpeg"
    
# #     if os.path.exists(test_image_path):
# #         image_path = test_image_path
# #         print("Using provided green circle OMR image")
# #     elif os.path.exists(fallback_image):
# #         image_path = fallback_image
# #         print("Using fallback image from dataset")
# #     else:
# #         print("No test image available")
# #         return
    
# #     # Process the image
# #     results = test_green_circle_omr(image_path)
    
# #     if results:
# #         # Create detailed analysis
# #         output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
# #         create_detailed_analysis_image(image_path, output_dir)
        
# #         print(f"\n🎉 GREEN CIRCLE OMR PROCESSING COMPLETE!")
# #         print(f"📁 All files saved to: {output_dir}")
# #         print(f"\nThe system is now optimized for green circle OMR sheets!")
# #         print(f"It can detect:")
# #         print(f"  ✅ Green circular answer bubbles")
# #         print(f"  ✅ Black filled marks inside circles") 
# #         print(f"  ✅ Multiple marked answers")
# #         print(f"  ✅ Unattempted questions")
# #         print(f"  ✅ Answer comparison with answer key")

# # if __name__ == "__main__":
# #     main()


























# # """
# # Green Circle OMR Processing Test
# # Specifically designed for the green circle OMR format with black filled marks
# # """

# # import os
# # import cv2
# # import json
# # import numpy as np
# # from enhanced_perfect_omr_debug import PerfectOMRDebugger

# # def test_green_circle_omr_with_custom_thresholds(image_path, fill_threshold=0.15, dark_threshold=150):
# #     """Test the green circle OMR processing with custom thresholds"""
    
# #     print("=" * 70)
# #     print("GREEN CIRCLE OMR PROCESSING TEST - 100 QUESTIONS")
# #     print(f"🎯 Fill Threshold: {fill_threshold} (Lower = More Sensitive)")
# #     print(f"🎯 Dark Threshold: {dark_threshold} (Higher = More Sensitive)")
# #     print("=" * 70)
    
# #     if not os.path.exists(image_path):
# #         print(f"Error: Image not found: {image_path}")
# #         return
    
# #     # Create output directory
# #     output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     print(f"Processing: {image_path}")
# #     print(f"Output directory: {output_dir}")
    
# #     # Initialize the debugger
# #     debugger = PerfectOMRDebugger()
    
# #     # Override the thresholds in the debugger if possible
# #     if hasattr(debugger, 'fill_threshold'):
# #         debugger.fill_threshold = fill_threshold
# #         print(f"✅ Set fill threshold to: {fill_threshold}")
# #     else:
# #         print(f"⚠️  Cannot set fill threshold - using default")
    
# #     if hasattr(debugger, 'dark_threshold'):
# #         debugger.dark_threshold = dark_threshold
# #         print(f"✅ Set dark threshold to: {dark_threshold}")
# #     else:
# #         print(f"⚠️  Cannot set dark threshold - using default")
    
# #     # Complete answer key for all 100 questions
# #     sample_answer_key = {
# #         'Set_A': {
# #             # Column 1: PYTHON (Questions 1-20)
# #             '1': ['a'], '2': ['c'], '3': ['c'], '4': ['c'], '5': ['c'],
# #             '6': ['a'], '7': ['c'], '8': ['c'], '9': ['b'], '10': ['c'],
# #             '11': ['a'], '12': ['a'], '13': ['d'], '14': ['a'], '15': ['a'],
# #             '16': ['b'], '17': ['a'], '18': ['d'], '19': ['a'], '20': ['b'],
            
# #             # Column 2: DATA ANALYSIS (Questions 21-40)
# #             '21': ['c'], '22': ['b'], '23': ['c'], '24': ['c'], '25': ['c'],
# #             '26': ['a'], '27': ['c'], '28': ['c'], '29': ['b'], '30': ['c'],
# #             '31': ['c'], '32': ['b'], '33': ['b'], '34': ['a'], '35': ['a'],
# #             '36': ['d'], '37': ['b'], '38': ['b'], '39': ['a'], '40': ['a'],
            
# #             # Column 3: MySQL (Questions 41-60)
# #             '41': ['c'], '42': ['a'], '43': ['c'], '44': ['a'], '45': ['a'],
# #             '46': ['c'], '47': ['a'], '48': ['c'], '49': ['a'], '50': ['d'],
# #             '51': ['c'], '52': ['c'], '53': ['b'], '54': ['c'], '55': ['c'],
# #             '56': ['a'], '57': ['a'], '58': ['c'], '59': ['a'], '60': ['c'],
            
# #             # Column 4: POWER BI (Questions 61-80)
# #             '61': ['c'], '62': ['c'], '63': ['c'], '64': ['d'], '65': ['d'],
# #             '66': ['d'], '67': ['c'], '68': ['a'], '69': ['c'], '70': ['a'],
# #             '71': ['c'], '72': ['c'], '73': ['b'], '74': ['c'], '75': ['c'],
# #             '76': ['c'], '77': ['c'], '78': ['c'], '79': ['c'], '80': ['c'],
            
# #             # Column 5: Adv STATS (Questions 81-100)
# #             '81': ['d'], '82': ['c'], '83': ['d'], '84': ['d'], '85': ['d'],
# #             '86': ['c'], '87': ['c'], '88': ['c'], '89': ['c'], '90': ['d'],
# #             '91': ['d'], '92': ['d'], '93': ['d'], '94': ['c'], '95': ['c'],
# #             '96': ['c'], '97': ['d'], '98': ['d'], '99': ['c'], '100': ['c']
# #         }
# #     }
    
# #     try:
# #         # Process the image with 100 questions and 4 choices
# #         print("\nStep 1: Reading and analyzing image...")
# #         results, final_image = debugger.process_omr_sheet(image_path, sample_answer_key, questions=100, choices=4)
        
# #         if results:
# #             # Save results with threshold info
# #             image_name = os.path.basename(image_path)
# #             base_name = os.path.splitext(image_name)[0]
            
# #             # Save JSON results
# #             results_path = os.path.join(output_dir, f"{base_name}_fill{fill_threshold}_dark{dark_threshold}_results.json")
            
# #             # Add threshold info to results
# #             results['processing_parameters'] = {
# #                 'fill_threshold': fill_threshold,
# #                 'dark_threshold': dark_threshold,
# #                 'total_questions': 100,
# #                 'columns': 5,
# #                 'questions_per_column': 20
# #             }
            
# #             with open(results_path, 'w') as f:
# #                 json.dump(results, f, indent=2)
            
# #             # Save processed image
# #             final_image_path = os.path.join(output_dir, f"{base_name}_fill{fill_threshold}_dark{dark_threshold}_processed.png")
# #             cv2.imwrite(final_image_path, final_image)
            
# #             # Save debug images
# #             debugger.save_debug_images(output_dir, f"{image_name}_fill{fill_threshold}_dark{dark_threshold}")
            
# #             print(f"\n✅ PROCESSING SUCCESSFUL!")
# #             print(f"Results saved to: {results_path}")
# #             print(f"Processed image saved to: {final_image_path}")
            
# #             # Display results summary
# #             print(f"\n📊 RESULTS SUMMARY:")
# #             print(f"Parameters: Fill={fill_threshold}, Dark={dark_threshold}")
# #             print(f"Source Image: {results['source_image']}")
# #             print(f"Detected Set: {results['detected_set_code']}")
# #             print(f"Total Score: {results['total_score']}/{results['max_score']}")
# #             print(f"Percentage: {results['percentage']}%")
            
# #             summary = results['summary']
# #             print(f"\nAnswer Breakdown:")
# #             print(f"✅ Correct: {summary['correct']}")
# #             print(f"❌ Incorrect: {summary['incorrect']}")
# #             print(f"⭕ Unattempted: {summary['unattempted']}")
# #             print(f"🔄 Multiple Marked: {summary['multiple_marked']}")
            
# #             # Show column-wise breakdown
# #             print(f"\n📝 Column-wise Analysis:")
# #             columns = {
# #                 'PYTHON': list(range(1, 21)),
# #                 'DATA ANALYSIS': list(range(21, 41)),
# #                 'MySQL': list(range(41, 61)),
# #                 'POWER BI': list(range(61, 81)),
# #                 'Adv STATS': list(range(81, 101))
# #             }
            
# #             for col_name, questions in columns.items():
# #                 col_correct = sum(1 for d in results['details'] if d['question'] in questions and d['status'] == 'Correct')
# #                 col_attempted = sum(1 for d in results['details'] if d['question'] in questions and d['marked'] != 'NONE')
# #                 col_percentage = (col_correct / len(questions)) * 100
                
# #                 print(f"{col_name:12}: {col_correct:2d}/20 correct ({col_percentage:5.1f}%), {col_attempted:2d}/20 attempted")
            
# #             # Show first 20 detailed results as sample
# #             print(f"\n📝 Sample Question Analysis (First 20):")
# #             print("Q#  | Marked | Correct | Status")
# #             print("-" * 35)
            
# #             for detail in results['details'][:20]:
# #                 q_num = f"Q{detail['question']:2d}"
# #                 marked = detail['marked'][:6] if detail['marked'] != 'NONE' else 'NONE  '
# #                 correct = ','.join(detail['correct'])[:6]
# #                 status = detail['status'][:12]
                
# #                 # Add status emoji
# #                 status_emoji = {
# #                     'Correct': '✅',
# #                     'Incorrect': '❌',
# #                     'Unattempted': '⭕',
# #                     'Multiple Marked': '🔄'
# #                 }.get(status, '❓')
                
# #                 print(f"{q_num} | {marked:6} | {correct:6} | {status_emoji} {status}")
            
# #             return results
            
# #         else:
# #             print("❌ Processing failed - no results returned")
# #             return None
            
# #     except Exception as e:
# #         print(f"❌ ERROR during processing: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         return None

# # def test_multiple_fill_thresholds(image_path):
# #     """Test with different fill ratio thresholds to find the best one"""
    
# #     print("\n🔧 TESTING MULTIPLE FILL RATIO THRESHOLDS (100 Questions)")
# #     print("=" * 60)
# #     print("Lower values = More sensitive (detect lighter marks)")
# #     print("Higher values = Less sensitive (only dark marks)")
# #     print("=" * 60)
    
# #     # Test different fill thresholds
# #     thresholds_to_test = [
# #         (0.10, 140),  # Very sensitive
# #         (0.15, 140),  # Sensitive  
# #         (0.20, 140),  # Moderate
# #         (0.25, 140),  # Conservative
# #         (0.30, 140),  # Default (likely too strict)
# #     ]
    
# #     results_comparison = []
    
# #     for fill_thresh, dark_thresh in thresholds_to_test:
# #         print(f"\n--- Testing Fill Threshold: {fill_thresh} ---")
# #         results = test_green_circle_omr_with_custom_thresholds(image_path, fill_thresh, dark_thresh)
        
# #         if results:
# #             detected_answers = len([d for d in results['details'] if d['marked'] != 'NONE'])
# #             score = results['total_score']
            
# #             results_comparison.append({
# #                 'fill_threshold': fill_thresh,
# #                 'detected_answers': detected_answers,
# #                 'score': score,
# #                 'percentage': results['percentage']
# #             })
            
# #             print(f"✅ Detected {detected_answers}/100 answers, Score: {score}/100 ({results['percentage']}%)")
# #         else:
# #             print(f"❌ Failed with threshold {fill_thresh}")
    
# #     # Show comparison
# #     if results_comparison:
# #         print(f"\n📊 FILL THRESHOLD COMPARISON (100 Questions)")
# #         print("=" * 55)
# #         print("Threshold | Detected | Score | Percentage")
# #         print("-" * 45)
        
# #         best_result = max(results_comparison, key=lambda x: x['detected_answers'])
        
# #         for result in results_comparison:
# #             marker = "🎯" if result == best_result else "  "
# #             print(f"{marker} {result['fill_threshold']:6.2f} | {result['detected_answers']:8d} | {result['score']:5d} | {result['percentage']:9.1f}%")
        
# #         print(f"\n🏆 RECOMMENDATION:")
# #         print(f"Best fill threshold: {best_result['fill_threshold']}")
# #         print(f"Detected {best_result['detected_answers']}/100 answers")
# #         print(f"Score: {best_result['score']}/100 ({best_result['percentage']}%)")
        
# #         return best_result
    
# #     return None
# # def create_detailed_analysis_image(image_path, output_dir):
# #     """Create a detailed analysis image showing the detection process"""
    
# #     print(f"\n🔍 Creating detailed analysis image...")
    
# #     # Read the image
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         print("Could not read image for analysis")
# #         return
    
# #     debugger = PerfectOMRDebugger()
    
# #     # Detect green circles and marks
# #     green_circles, marked_circles = debugger.detect_green_circles_and_black_marks(img)
    
# #     # Create analysis image
# #     analysis_img = img.copy()
    
# #     # Draw all green circles in green
# #     for i, circle in enumerate(green_circles):
# #         center = circle['center']
# #         radius = circle['radius']
# #         cv2.circle(analysis_img, center, radius, (0, 255, 0), 2)
# #         cv2.circle(analysis_img, center, 3, (0, 255, 0), -1)
        
# #         # Add circle number
# #         cv2.putText(analysis_img, str(i+1), 
# #                    (center[0] - 10, center[1] - radius - 10), 
# #                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
# #     # Draw marked circles in red with fill ratios
# #     for i, circle in enumerate(marked_circles):
# #         center = circle['center']
# #         radius = circle['radius']
# #         fill_ratio = circle['fill_ratio']
        
# #         cv2.circle(analysis_img, center, radius, (0, 0, 255), 3)
# #         cv2.circle(analysis_img, center, 5, (0, 0, 255), -1)
        
# #         # Add fill ratio text
# #         cv2.putText(analysis_img, f"{fill_ratio:.2f}", 
# #                    (center[0] + radius + 5, center[1]), 
# #                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
# #     # Add summary text
# #     cv2.putText(analysis_img, f"Green Circles: {len(green_circles)}", 
# #                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #     cv2.putText(analysis_img, f"Marked Circles: {len(marked_circles)}", 
# #                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
# #     # Save analysis image
# #     analysis_path = os.path.join(output_dir, "detailed_analysis.png")
# #     cv2.imwrite(analysis_path, analysis_img)
# #     print(f"Analysis image saved to: {analysis_path}")
    
# #     return len(green_circles), len(marked_circles)

# # def main():
# #     """Main function to test green circle OMR processing with different thresholds"""
    
# #     # Test with the provided image (save it first)
# #     test_image_path = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\test_green_circle_omr.png"
    
# #     # For now, we'll test with an existing image from the dataset
# #     fallback_image = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data\Set_B\Img12.jpeg"
    
# #     if os.path.exists(test_image_path):
# #         image_path = test_image_path
# #         print("Using provided green circle OMR image")
# #     elif os.path.exists(fallback_image):
# #         image_path = fallback_image
# #         print("Using fallback image from dataset")
# #     else:
# #         print("No test image available")
# #         return
    
# #     # Create detailed analysis first
# #     output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
# #     green_count, marked_count = create_detailed_analysis_image(image_path, output_dir)
    
# #     print(f"\n📊 Initial Circle Detection:")
# #     print(f"Green Circles Found: {green_count}")
# #     print(f"Marked Circles Found: {marked_count}")
    
# #     # Test multiple thresholds to find the best one
# #     best_settings = test_multiple_fill_thresholds(image_path)
    
# #     if best_settings:
# #         print(f"\n🎉 TESTING COMPLETE!")
# #         print(f"📁 All results saved to: {output_dir}")
# #         print(f"\n💡 FINAL RECOMMENDATION:")
# #         print(f"  🎯 Use Fill Threshold: {best_settings['fill_threshold']}")
# #         print(f"  📈 This detected {best_settings['detected_answers']} answers")
# #         print(f"  🏆 Score: {best_settings['score']} ({best_settings['percentage']}%)")
        
# #         print(f"\nTo use these settings permanently, modify your PerfectOMRDebugger class:")
# #         print(f"  Change fill_ratio threshold from 0.3 to {best_settings['fill_threshold']}")

# # if __name__ == "__main__":
# #     main()





















# """
# Fully dynamic Green Circle OMR Processor
# No hard‑coded columns, set names, question count, choices or workbook path.
# Automatically:
#  - Finds first Excel workbook in dataset/data (or one containing 'key')
#  - Reads every sheet (each sheet = one set)
#  - Infers Question / Answer / Subject columns heuristically
#  - Builds answer key + metadata
#  - Detects set via OCR (fallback to sheet order)
#  - Calls PerfectOMRDebugger once
#  - Re-scores using Excel key
# Place workbook in: ../dataset/data/  (e.g. 'Key (Set A and B).xlsx')
# """

# import os, re, json, cv2
# from datetime import datetime
# from collections import defaultdict, Counter
# from enhanced_perfect_omr_debug import PerfectOMRDebugger

# try:
#     import pandas as pd
# except ImportError:
#     pd = None
# try:
#     import pytesseract
# except ImportError:
#     pytesseract = None

# # ----------------- Workbook Discovery -----------------
# def find_answer_key_workbook(root_dir):
#     excel_files = []
#     for f in os.listdir(root_dir):
#         if f.lower().endswith((".xlsx", ".xls")):
#             excel_files.append(f)
#     if not excel_files:
#         raise FileNotFoundError("No Excel workbook found in dataset/data")
#     # Prefer file containing 'key'
#     for f in excel_files:
#         if 'key' in f.lower():
#             return os.path.join(root_dir, f)
#     return os.path.join(root_dir, excel_files[0])

# # ----------------- Flexible Sheet Parsing -----------------
# def infer_columns(df):
#     """
#     Heuristically infer question, answer, subject columns.
#     Returns (q_col, a_col, subj_col or None)
#     """
#     cols = list(df.columns)
#     cleaned = {c: re.sub(r'[\s\.\-_]+','', str(c).lower()) for c in cols}

#     # Direct name hints
#     q_hints = {'q','qn','qno','question','ques','questionno','qnumber'}
#     a_hints = {'ans','answer','correct','key','correctoption','correctoptions','solution','resp','responses'}
#     s_hints = {'subject','sub','topic','section','language','lang','area','domain'}

#     q_col = next((c for c in cols if cleaned[c] in q_hints), None)
#     a_col = next((c for c in cols if cleaned[c] in a_hints), None)
#     subj_col = next((c for c in cols if cleaned[c] in s_hints), None)

#     # If still missing, analyze content
#     def score_question(col):
#         series = df[col].dropna().astype(str).head(50)
#         if series.empty: return 0
#         m_digit = sum(bool(re.match(r'^\s*\d+\s*$', s)) for s in series)
#         m_qdigit = sum(bool(re.match(r'^\s*(Q|q)\s*\d+\s*$', s)) for s in series)
#         return (m_digit + m_qdigit) / len(series)

#     def score_answer(col):
#         series = df[col].dropna().astype(str).head(50)
#         if series.empty: return 0
#         pat = re.compile(r'^[A-Za-z]{1,5}([,/ ][A-Za-z]{1,5})*$')  # combinations
#         short = sum(len(s) <= 10 for s in series)
#         match = sum(bool(pat.match(s.replace('-', '').replace(';',' '))) for s in series)
#         return (match + short*0.3)/len(series)

#     if q_col is None:
#         scores = {c: score_question(c) for c in cols}
#         q_col = max(scores, key=scores.get) if scores and scores[max(scores, key=scores.get)] > 0 else None
#     if a_col is None:
#         scores = {c: score_answer(c) for c in cols if c != q_col}
#         a_col = max(scores, key=scores.get) if scores and scores[max(scores, key=scores.get)] > 0 else None

#     # Subject inference: pick texty column with few unique values not used
#     if subj_col is None:
#         best_col = None
#         best_score = 0
#         for c in cols:
#             if c in (q_col, a_col): continue
#             series = df[c].dropna().astype(str)
#             if series.empty: continue
#             uniq = series.nunique()
#             avg_len = sum(len(s) for s in series[:50]) / min(len(series),50)
#             if 1 < uniq < 30 and avg_len > 2:
#                 density = uniq / len(series)
#                 score = (1 - density) + (avg_len/50)
#                 if score > best_score:
#                     best_score = score
#                     best_col = c
#         subj_col = best_col

#     return q_col, a_col, subj_col

# def normalize_set_name(sheet_name):
#     m = re.search(r'([A-Z])', sheet_name.upper())
#     letter = m.group(1) if m else sheet_name[:1].upper()
#     return f"Set_{letter}"

# def _parse_wide_subject_sheet(df, sheet_name):
#     """
#     Fallback for sheets where each column is a subject and rows are answers.
#     No explicit question numbers present.
#     """
#     set_code = normalize_set_name(sheet_name)
#     answer_block = {}
#     meta_block = {"subjects": set(), "max_q": 0, "choices": set()}

#     question_counter = 0
#     for col in df.columns:
#         subj_raw = str(col).strip()
#         if not subj_raw:
#             continue
#         subject_name = subj_raw  # keep original (can normalize if needed)
#         column_series = df[col]

#         # Collect answers in order
#         for cell in column_series:
#             if pd.isna(cell):
#                 continue
#             ans_raw = str(cell).strip()
#             if not ans_raw or ans_raw.lower() in ("nan",):
#                 continue

#             # Heuristic: must look like answer tokens (letters / delimiters)
#             if not re.fullmatch(r'[A-Za-z ,;/.-]+', ans_raw):
#                 # If cell contains long unexpected text, skip
#                 if len(ans_raw) > 12:
#                     continue

#             question_counter += 1

#             ans_clean = ans_raw.replace('-', ' ').replace('.', ' ').replace(';', ' ').strip()
#             if ',' in ans_clean:
#                 tokens = [t.strip().lower() for t in ans_clean.split(',') if t.strip()]
#             elif re.fullmatch(r'[A-Za-z]+', ans_clean) and len(ans_clean) > 1:
#                 tokens = [c.lower() for c in ans_clean]
#             else:
#                 tokens = [t.lower() for t in re.split(r'[\s/]+', ans_clean) if t]

#             if not tokens:
#                 continue

#             q_num = str(question_counter)
#             answer_block[q_num] = {"options": tokens, "subject": subject_name}
#             meta_block["subjects"].add(subject_name)
#             meta_block["max_q"] = max(meta_block["max_q"], question_counter)
#             meta_block["choices"].update(tokens)

#     if question_counter == 0:
#         return None, None

#     return {set_code: answer_block}, {set_code: meta_block}

# def parse_workbook(excel_path):
#     """
#     Tries structured parsing first (question/answer columns). If that fails,
#     uses wide-subject fallback where each column is a subject.
#     """
#     if pd is None:
#         raise ImportError("pandas required")
#     xl = pd.ExcelFile(excel_path)
#     full_answer_key = {}
#     full_meta = {}

#     for sheet in xl.sheet_names:
#         df = pd.read_excel(excel_path, sheet_name=sheet)
#         if df.empty:
#             continue

#         # Attempt structured inference
#         q_col, a_col, subj_col = infer_columns(df)

#         if q_col and a_col:
#             set_code = normalize_set_name(sheet)
#             full_answer_key.setdefault(set_code, {})
#             full_meta.setdefault(set_code, {"subjects": set(), "max_q":0, "choices": set()})
#             for _, row in df.iterrows():
#                 q_raw = str(row.get(q_col, '')).strip()
#                 if not q_raw:
#                     continue
#                 mnum = re.search(r'\d+', q_raw)
#                 if not mnum:
#                     continue
#                 q_num = mnum.group(0)

#                 ans_raw = str(row.get(a_col, '')).strip()
#                 if not ans_raw:
#                     continue

#                 ans_clean = ans_raw.replace('-', ' ').replace(';', ' ').replace('.', ' ').strip()
#                 if ',' in ans_clean:
#                     tokens = [t.strip().lower() for t in ans_clean.split(',') if t.strip()]
#                 elif re.fullmatch(r'[A-Za-z]+', ans_clean) and len(ans_clean) > 1:
#                     tokens = [c.lower() for c in ans_clean]
#                 else:
#                     tokens = [t.lower() for t in re.split(r'[\s/]+', ans_clean) if t]

#                 if not tokens:
#                     continue

#                 subj_val = "GENERAL"
#                 if subj_col and not pd.isna(row.get(subj_col)):
#                     subj_val = str(row.get(subj_col)).strip() or "GENERAL"

#                 full_answer_key[set_code][q_num] = {"options": tokens, "subject": subj_val}
#                 mdata = full_meta[set_code]
#                 mdata["subjects"].add(subj_val)
#                 mdata["max_q"] = max(mdata["max_q"], int(q_num))
#                 mdata["choices"].update(tokens)

#             print(f"Parsed structured sheet: {sheet} -> {set_code} (questions={full_meta[set_code]['max_q']})")
#             continue

#         # Fallback: wide subject sheet
#         wide_key, wide_meta = _parse_wide_subject_sheet(df, sheet)
#         if wide_key:
#             set_code = list(wide_key.keys())[0]
#             full_answer_key.update(wide_key)
#             full_meta.update(wide_meta)
#             print(f"Parsed wide subject sheet: {sheet} -> {set_code} (questions={wide_meta[set_code]['max_q']})")
#         else:
#             print(f"Skip sheet {sheet}: could not parse (columns={df.columns.tolist()})")

#     if not full_answer_key:
#         raise ValueError("No usable sheets parsed. Check workbook format.")
#     return full_answer_key, full_meta



# # ...rest of existing code (ocr_set_code, score_details, subject_scores, DynamicOMRProcessor, main) remains unchanged...
# # ----------------- OCR Set Detection -----------------
# def ocr_set_code(image):
#     if pytesseract is None:
#         return None
#     h,w = image.shape[:2]
#     roi = image[:int(0.18*h), :]  # broad top band
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#     txt = pytesseract.image_to_string(gray, config="--psm 6").upper()
#     m = re.search(r"SET[\s:_-]*([A-Z])", txt)
#     if m: return f"Set_{m.group(1)}"
#     letters = re.findall(r"\b([A-Z])\b", txt)
#     if letters: return f"Set_{letters[0]}"
#     return None
# import json, os
# from collections import Counter
# from datetime import datetime

# EVAL_ORDER = ["correct","incorrect","multiple_marked","partially_marked","unattempted","no_key"]

# def evaluate_answers(detected_answers, answer_key, sheet_set_code="Set_A",
#                      source_image="", extra_meta=None, out_path=None):
#     """
#     detected_answers: list of dicts: { "question": int, "marked": "a,b" | "NONE" }
#     answer_key: { "1": {"options":["a","c"], "subject":"PYTHON"}, ... }
#     """
#     summary = Counter()
#     subject_correct = Counter()
#     details_out = []

#     for item in detected_answers:
#         q_str = str(item["question"])
#         ak = answer_key.get(q_str)
#         marked_field = item.get("marked","NONE")
#         if not ak:
#             status = "NoKey"
#             summary["no_key"] += 1
#             correct_opts = []
#         else:
#             correct_opts = ak["options"]
#             corr_set = set(correct_opts)
#             if marked_field == "NONE":
#                 status = "Unattempted"
#                 summary["unattempted"] += 1
#             else:
#                 marked_set = set(m.strip().lower() for m in marked_field.split(",") if m.strip())
#                 if len(marked_set) > 1 and marked_set != corr_set:
#                     status = "Multiple Marked"
#                     summary["multiple_marked"] += 1
#                 elif marked_set == corr_set:
#                     status = "Correct"
#                     summary["correct"] += 1
#                     subject_correct[ak["subject"]] += 1
#                 elif marked_set.issubset(corr_set):
#                     status = "Partially Marked"
#                     summary["partially_marked"] += 1
#                 else:
#                     status = "Incorrect"
#                     summary["incorrect"] += 1
#         details_out.append({
#             "question": item["question"],
#             "marked": marked_field,
#             "correct": ",".join(correct_opts),
#             "status": status
#         })

#     # Ensure all keys exist
#     for k in ["correct","incorrect","multiple_marked","partially_marked","unattempted","no_key"]:
#         summary[k] = summary.get(k,0)

#     total_questions = len(detected_answers)
#     total_score = summary["correct"]
#     max_score = sum(1 for q in detected_answers if str(q["question"]) in answer_key)

#     report = {
#         "exam_set": sheet_set_code,
#         "source_image": source_image,
#         "total_score": total_score,
#         "max_score": max_score,
#         "percentage": round(100*total_score/max_score,2) if max_score else 0.0,
#         "summary": {k: summary[k] for k in EVAL_ORDER},
#         "subject_scores": dict(subject_correct),
#         "details": details_out,
#         "metadata": {
#             "questions_detected": total_questions,
#             "questions_with_key": max_score,
#             **(extra_meta or {})
#         },
#         "timestamp": datetime.utcnow().isoformat()
#     }

#     if out_path:
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         with open(out_path,"w") as f:
#             json.dump(report, f, indent=2)
#     return report

# # Helper to adapt a simple raw key format if needed
# def normalize_simple_key(raw_key):
#     """
#     raw_key examples:
#       { "1":"a", "2":"b", "3":"ac" }  or  { "1":["a"], "2":["b"], "3":["a","c"] }
#     Returns structured dict with subject default 'GENERAL'.
#     """
#     norm = {}
#     for q, v in raw_key.items():
#         if isinstance(v, str):
#             opts = list(v.lower())
#         else:
#             opts = [o.lower() for o in v]
#         norm[str(q)] = {"options": opts, "subject":"GENERAL"}
#     return norm
# # ----------------- Scoring -----------------
# def score_details(details, key_set):
#     summary = Counter()
#     for d in details:
#         q = str(d["question"])
#         entry = key_set.get(q)
#         if not entry:
#             d["status"]="NoKey"
#             continue
#         correct = set(entry["options"])
#         if d["marked"]=="NONE":
#             d["status"]="Unattempted"; summary["unattempted"]+=1; continue
#         marked = set([x.strip().lower() for x in d["marked"].split(",")])
#         if len(marked)>1 and marked!=correct:
#             d["status"]="Multiple Marked"; summary["multiple_marked"]+=1; continue
#         if marked==correct:
#             d["status"]="Correct"; summary["correct"]+=1
#         elif marked.issubset(correct):
#             d["status"]="Partially Marked"; summary["partially_marked"]+=1
#         else:
#             d["status"]="Incorrect"; summary["incorrect"]+=1
#     for k in ["correct","incorrect","unattempted","multiple_marked","partially_marked"]:
#         summary[k]=summary.get(k,0)
#     return dict(summary)

# def subject_scores(details, key_set):
#     subs = Counter()
#     for d in details:
#         e = key_set.get(str(d["question"]))
#         if e and d["status"]=="Correct":
#             subs[e["subject"]]+=1
#     return dict(subs)

# # ----------------- Processor -----------------
# class DynamicOMRProcessor:
#     def __init__(self, excel_path):
#         self.answer_key, self.meta = parse_workbook(excel_path)
#         self.excel_path = excel_path
#         self.debugger = PerfectOMRDebugger()

#     def select_set(self, img):
#         detected = ocr_set_code(img)
#         if detected and detected in self.answer_key:
#             return detected
#         # fallback first
#         return list(self.answer_key.keys())[0]

#     def process(self, image_path, output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#         img = cv2.imread(image_path)
#         if img is None: raise ValueError("Cannot read image")
#         set_code = self.select_set(img)
#         key_set = self.answer_key[set_code]
#         total_q = self.meta[set_code]["max_q"]
#         choices_cnt = max(len(self.meta[set_code]["choices"]), 4)  # debugger needs min
#         legacy = {set_code: {q: v["options"] for q,v in key_set.items()}}
#         results, annotated = self.debugger.process_omr_sheet(
#             image_path, legacy, questions=total_q, choices=choices_cnt
#         )
#         if not results: raise RuntimeError("Debugger returned no results")
#         summary = score_details(results["details"], key_set)
#         subj = subject_scores(results["details"], key_set)
#         total_score = summary["correct"]
#         report = {
#             "source_image": os.path.basename(image_path),
#             "exam_set": set_code,
#             "total_score": total_score,
#             "max_score": total_q,
#             "percentage": round(100*total_score/total_q,2),
#             "subject_scores": subj,
#             "summary": summary,
#             "details": results["details"],
#             "questions_with_answer_key": len(key_set),
#             "detected_set_code": results.get("detected_set_code"),
#             "processing_parameters": {
#                 "inferred_total_questions": total_q,
#                 "inferred_choices": choices_cnt,
#                 "excel_path": self.excel_path
#             },
#             "timestamp": datetime.utcnow().isoformat()
#         }
#         base = os.path.splitext(os.path.basename(image_path))[0]
#         with open(os.path.join(output_dir, f"{base}_dynamic.json"), "w") as f:
#             json.dump(report, f, indent=2)
#         cv2.imwrite(os.path.join(output_dir, f"{base}_dynamic_annotated.png"), annotated)
#         self.debugger.save_debug_images(output_dir, base+"_dyn")
#         print(f"Saved report for {set_code}: {report['total_score']}/{report['max_score']}")
#         return report

# # ----------------- CLI -----------------
# def main():
#     backend_dir = os.path.dirname(__file__)
#     data_dir = os.path.join(backend_dir, "..", "dataset", "data")
#     try:
#         excel_path = find_answer_key_workbook(data_dir)
#     except Exception as e:
#         print(f"Workbook error: {e}")
#         return
#     # pick first image automatically
#     image_path = None
#     for dp,_,files in os.walk(data_dir):
#         for f in files:
#             if f.lower().endswith(('.jpg','.jpeg','.png')):
#                 image_path = os.path.join(dp,f)
#                 break
#         if image_path: break
#     if not image_path:
#         print("No image found.")
#         return
#     out_dir = os.path.join(backend_dir, "green_circle_output")
#     proc = DynamicOMRProcessor(excel_path)
#     proc.process(image_path, out_dir)

# if __name__ == "__main__":
#     main()