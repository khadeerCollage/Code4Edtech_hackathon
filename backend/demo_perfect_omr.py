"""
Perfect OMR System Demonstration
Shows how to use the enhanced OMR system with real images from the dataset
"""

import os
import json
import cv2
from enhanced_perfect_omr_debug import PerfectOMRDebugger, process_single_image

def process_green_circle_omr():
    """Process green circle OMR format like the provided image"""
    
    print(f"\n{'='*70}")
    print("GREEN CIRCLE OMR PROCESSING - OPTIMIZED FOR YOUR IMAGE FORMAT")
    print(f"{'='*70}")
    
    # Test image paths
    dataset_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data"
    set_a_dir = os.path.join(dataset_dir, "Set_A")
    output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\green_circle_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample answer key matching the green circle format
    green_circle_answer_key = {
        'Set_A': {
            '1': ['a'], '2': ['b'], '3': ['c'], '4': ['d'], '5': ['a'],
            '6': ['b'], '7': ['c'], '8': ['d'], '9': ['a'], '10': ['b'],
            '11': ['c'], '12': ['d'], '13': ['a'], '14': ['b'], '15': ['c'],
            '16': ['d'], '17': ['a'], '18': ['b'], '19': ['c'], '20': ['d']
        }
    }
    
    print("🎯 Processing OMR sheets with GREEN CIRCLE format:")
    print("   ✅ Detects green circular answer bubbles")
    print("   ✅ Identifies black filled marks inside circles")
    print("   ✅ Organizes circles into question-answer grid")
    print("   ✅ Compares with answer key for scoring")
    
    # Get test images
    if os.path.exists(set_a_dir):
        images = [f for f in os.listdir(set_a_dir) if f.endswith('.jpeg')][:2]  # Test 2 images
        
        results_summary = []
        
        for i, image_name in enumerate(images, 1):
            image_path = os.path.join(set_a_dir, image_name)
            
            print(f"\n{'='*20} Processing Image {i} {'='*20}")
            print(f"Image: {image_name}")
            
            try:
                # Create debugger instance
                debugger = PerfectOMRDebugger()
                
                # Process with green circle optimized method
                results, final_image = debugger.process_omr_sheet(
                    image_path, green_circle_answer_key, questions=20, choices=4
                )
                
                if results:
                    # Save results
                    base_name = os.path.splitext(image_name)[0]
                    
                    # Save JSON
                    results_path = os.path.join(output_dir, f"{base_name}_green_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Save processed image
                    final_path = os.path.join(output_dir, f"{base_name}_green_processed.png")
                    cv2.imwrite(final_path, final_image)
                    
                    # Create analysis summary
                    summary = {
                        'image': image_name,
                        'score': f"{results['total_score']}/{results['max_score']}",
                        'percentage': f"{results['percentage']}%",
                        'correct': results['summary']['correct'],
                        'incorrect': results['summary']['incorrect'],
                        'unattempted': results['summary']['unattempted'],
                        'multiple_marked': results['summary']['multiple_marked']
                    }
                    
                    results_summary.append(summary)
                    
                    print(f"✅ SUCCESS: {summary['score']} ({summary['percentage']})")
                    print(f"   📊 Correct: {summary['correct']}, Wrong: {summary['incorrect']}")
                    print(f"   📊 Empty: {summary['unattempted']}, Multiple: {summary['multiple_marked']}")
                    
                    # Show some detected answers
                    detected_answers = [detail['marked'] for detail in results['details'][:10]]
                    print(f"   🔍 First 10 answers: {detected_answers}")
                    
                else:
                    print(f"❌ Failed to process {image_name}")
                    
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
        
        # Summary
        if results_summary:
            print(f"\n{'='*50}")
            print("GREEN CIRCLE OMR PROCESSING SUMMARY")
            print(f"{'='*50}")
            
            print(f"Successfully processed {len(results_summary)} images:")
            for result in results_summary:
                print(f"📄 {result['image']}: {result['score']} ({result['percentage']})")
            
            print(f"\n📁 All results saved to: {output_dir}")
            
            # Show features
            print(f"\n🎯 GREEN CIRCLE OMR FEATURES:")
            print(f"   🟢 Green Circle Detection: HSV color filtering")
            print(f"   ⚫ Black Mark Detection: Intensity thresholding")
            print(f"   📐 Grid Organization: Spatial sorting algorithm")
            print(f"   🎯 Answer Matching: Distance-based circle matching")
            print(f"   📊 Scoring: Answer key comparison")
            print(f"   🖼️ Visualization: Color-coded result marking")
        
    else:
        print(f"❌ Dataset directory not found: {set_a_dir}")

def process_multiple_images():
    """Process multiple images from the dataset and show results"""
    
    # Setup paths
    dataset_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\dataset\data"
    set_a_dir = os.path.join(dataset_dir, "Set_A")
    excel_path = os.path.join(dataset_dir, "Key (Set A and B).xlsx")
    output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\output"
    
    print("=" * 70)
    print("PERFECT OMR SYSTEM - MULTIPLE IMAGE PROCESSING")
    print("=" * 70)
    
    # Get all images from Set A
    if os.path.exists(set_a_dir):
        images = [f for f in os.listdir(set_a_dir) if f.endswith('.jpeg')]
        images.sort()  # Sort for consistent processing order
        
        print(f"Found {len(images)} images in Set A:")
        for img in images:
            print(f"  - {img}")
        
        # Process first 3 images as demonstration
        demo_images = images[:3]
        
        print(f"\nProcessing {len(demo_images)} images for demonstration...")
        
        results_summary = []
        
        for i, image_name in enumerate(demo_images, 1):
            image_path = os.path.join(set_a_dir, image_name)
            
            print(f"\n{'='*20} Processing Image {i}/{len(demo_images)} {'='*20}")
            print(f"Image: {image_name}")
            
            try:
                # Process the image
                results = process_single_image(image_path, excel_path, output_dir)
                
                if results:
                    # Extract key information
                    summary = {
                        'image': image_name,
                        'set': results['detected_set_code'] or 'N/A',
                        'score': f"{results['total_score']}/{results['max_score']}",
                        'percentage': f"{results['percentage']}%",
                        'correct': results['summary']['correct'],
                        'incorrect': results['summary']['incorrect'],
                        'unattempted': results['summary']['unattempted'],
                        'multiple_marked': results['summary']['multiple_marked']
                    }
                    
                    results_summary.append(summary)
                    
                    print(f"✅ SUCCESS: {summary['score']} ({summary['percentage']})")
                    print(f"   Correct: {summary['correct']}, Incorrect: {summary['incorrect']}")
                    print(f"   Unattempted: {summary['unattempted']}, Multiple: {summary['multiple_marked']}")
                else:
                    print(f"❌ FAILED to process {image_name}")
                    
            except Exception as e:
                print(f"❌ ERROR processing {image_name}: {e}")
        
        # Print final summary
        print(f"\n{'='*50}")
        print("PROCESSING SUMMARY")
        print(f"{'='*50}")
        
        if results_summary:
            print(f"Successfully processed {len(results_summary)} images:")
            print(f"{'Image':<15} {'Set':<5} {'Score':<10} {'%':<8} {'Correct':<8} {'Wrong':<6} {'Empty':<6} {'Multi':<6}")
            print("-" * 70)
            
            for result in results_summary:
                print(f"{result['image']:<15} {result['set']:<5} {result['score']:<10} "
                      f"{result['percentage']:<8} {result['correct']:<8} {result['incorrect']:<6} "
                      f"{result['unattempted']:<6} {result['multiple_marked']:<6}")
            
            # Calculate overall statistics
            total_correct = sum(r['correct'] for r in results_summary)
            total_questions = len(results_summary) * 20  # Assuming 20 questions per image
            overall_percentage = (total_correct / total_questions) * 100
            
            print("-" * 70)
            print(f"Overall: {total_correct}/{total_questions} ({overall_percentage:.1f}%)")
        
        print(f"\n✅ All output files saved to: {output_dir}")
        
    else:
        print(f"❌ Dataset directory not found: {set_a_dir}")

def demonstrate_advanced_features():
    """Demonstrate advanced features of the OMR system"""
    
    print(f"\n{'='*50}")
    print("ADVANCED FEATURES DEMONSTRATION")
    print(f"{'='*50}")
    
    # Create a demonstration OMR debugger
    debugger = PerfectOMRDebugger()
    
    print("\n1. Advanced Thresholding Techniques:")
    print("   ✅ Global threshold detection using gap analysis")
    print("   ✅ Local threshold for individual question strips")
    print("   ✅ CLAHE enhancement for better contrast")
    print("   ✅ Gamma correction for image optimization")
    print("   ✅ Morphological operations for noise reduction")
    
    print("\n2. OCR Integration:")
    print("   ✅ Automatic Set A/B detection from image headers")
    print("   ✅ Multiple preprocessing strategies for better OCR")
    print("   ✅ Confidence-based text recognition")
    
    print("\n3. Answer Processing:")
    print("   ✅ Multiple marked answer detection")
    print("   ✅ Partially marked answer handling")
    print("   ✅ Unattempted question identification")
    print("   ✅ Answer key integration from Excel files")
    
    print("\n4. Debug and Visualization:")
    print("   ✅ Step-by-step processing visualization")
    print("   ✅ Debug images for each processing stage")
    print("   ✅ Comprehensive JSON output")
    print("   ✅ Error tracking and reporting")
    
    print("\n5. Output Format Compatibility:")
    print("   ✅ Matches test_Img5_detailed.json structure exactly")
    print("   ✅ Includes timestamp and metadata")
    print("   ✅ Subject-wise scoring capability")
    print("   ✅ Detection method tracking")
    
    # Show threshold parameters
    print(f"\n6. Threshold Parameters:")
    for param, value in debugger.threshold_params.items():
        print(f"   {param}: {value}")

def show_file_structure():
    """Show the output file structure"""
    
    print(f"\n{'='*50}")
    print("OUTPUT FILE STRUCTURE")
    print(f"{'='*50}")
    
    output_dir = r"C:\Users\USER\Desktop\Code4Edtech_hackathon\Code4Edtech_hackathon\backend\output"
    
    if os.path.exists(output_dir):
        print(f"Output Directory: {output_dir}")
        
        # List main files
        main_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        if main_files:
            print(f"\nMain Output Files ({len(main_files)}):")
            for file in sorted(main_files):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size:,} bytes)")
        
        # List debug directory
        debug_dir = os.path.join(output_dir, "debug")
        if os.path.exists(debug_dir):
            debug_files = [f for f in os.listdir(debug_dir) if f.endswith('.png')]
            if debug_files:
                print(f"\nDebug Images ({len(debug_files)}):")
                stages = {}
                for file in debug_files:
                    parts = file.split('_')
                    if len(parts) >= 2:
                        stage = parts[-1].replace('.png', '')
                        if stage not in stages:
                            stages[stage] = []
                        stages[stage].append(file)
                
                for stage, files in sorted(stages.items()):
                    print(f"  🖼️  {stage}: {len(files)} image(s)")
        
        print(f"\n📊 Sample JSON Structure:")
        # Try to load a sample JSON file
        json_files = [f for f in main_files if f.endswith('.json')]
        if json_files:
            sample_path = os.path.join(output_dir, json_files[0])
            try:
                with open(sample_path, 'r') as f:
                    sample_data = json.load(f)
                
                print(f"   Source: {sample_data.get('source_image', 'N/A')}")
                print(f"   Set: {sample_data.get('exam_set', 'N/A')}")
                print(f"   Score: {sample_data.get('total_score', 0)}/{sample_data.get('max_score', 0)}")
                print(f"   Questions: {len(sample_data.get('details', []))}")
                print(f"   Timestamp: {sample_data.get('timestamp', 'N/A')}")
                print(f"   Detection Method: {sample_data.get('detection_method', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ Error reading sample JSON: {e}")
    else:
        print(f"❌ Output directory not found: {output_dir}")

def main():
    """Main demonstration function"""
    
    print("🎯 PERFECT OMR SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    # Process green circle OMR format first (optimized for your image)
    process_green_circle_omr()
    
    # Process multiple images with original method
    process_multiple_images()
    
    # Show advanced features
    demonstrate_advanced_features()
    
    # Show file structure
    show_file_structure()
    
    print(f"\n{'='*80}")
    print("✅ DEMONSTRATION COMPLETE")
    print(f"{'='*80}")
    
    print(f"""
🚀 System Ready! You can now:

1. Process GREEN CIRCLE OMR images (optimized for your format):
   python test_green_circle_omr.py

2. Process single images:
   python enhanced_perfect_omr_debug.py image_path.jpg

3. Use in your code:
   from enhanced_perfect_omr_debug import process_single_image
   results = process_single_image(image_path, excel_path, output_dir)

4. Batch process images:
   python demo_perfect_omr.py

🎯 GREEN CIRCLE OMR FEATURES:
   ✅ Detects green circular answer bubbles
   ✅ Identifies black filled marks inside circles  
   ✅ Handles multiple marked answers
   ✅ Compares with answer key for accurate scoring
   ✅ Generates detailed JSON results matching your format

The system outputs match the exact format of test_Img5_detailed.json!
""")

if __name__ == "__main__":
    main()