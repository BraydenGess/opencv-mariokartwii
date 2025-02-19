# Usage: python3 test_course_detector.py [-h] [--method {cnn,template}] [--test_dir TEST_DIR] model_path
#
# Test course detector on validation data
#
# Positional arguments:
#   model_path            Path to trained flag detection model
#
# Optional arguments:
#   -h, --help           Show this help message and exit
#   --method {cnn,template}
#                        Detection method to use (default: cnn)
#   --test_dir TEST_DIR  Path to test data directory containing course subdirectories
#                        (default: Images/flag_data)


import cv2
from course_detector import CourseDetector
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
import time
from difflib import SequenceMatcher
import sys
import argparse

class CourseDetectorTester:
    def __init__(self, model_path, test_data_path, use_template=False):
        self.detector = CourseDetector(model_path, use_template=use_template)
        self.test_data_path = Path(test_data_path)
        self.known_courses = set(course.name for course in self.test_data_path.glob('*') if course.is_dir())
        
        # Create output directory for visualizations
        self.output_dir = Path('metrics/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def draw_detection_results(self, frame, flag_detected, flag_confidence, text_detections, matched_course):
        """Draw detection results on the frame"""
        vis_frame = frame.copy()
        
        # Draw flag ROI in blue
        x, y, w, h = self.detector.flag_roi
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle
        cv2.putText(vis_frame, "Flag ROI", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw all detected text
        for detection in text_detections:
            x, y, w, h = detection['box']
            text = detection['text']
            conf = detection['confidence']
            
            # Draw box around text
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            # Draw text and confidence
            cv2.putText(vis_frame, f"{text} ({conf}%)", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Draw course ROI in green
        x, y, w, h = self.detector.text_roi
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for course ROI
        cv2.putText(vis_frame, "Course ROI", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add flag detection result
        color = (0, 255, 0) if flag_detected else (0, 0, 255)  # Green if detected, Red if not
        cv2.putText(vis_frame, f"Flag: {flag_confidence:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add matched course if found
        if matched_course:
            cv2.putText(vis_frame, f"Course: {matched_course}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame
    
    def test_on_dataset(self):
        # Initialize metrics
        results = {
            'flag_detection': [],
            'ocr_results': [],
            'processing_times': [],
            'flag_confidences': [],
            'course_confidences': []
        }
        
        # Process each course directory
        for course_dir in self.test_data_path.glob('*'):
            if not course_dir.is_dir():
                continue
                
            expected_course = course_dir.name
            print(f"\nProcessing {expected_course}...")
            
            # Process each image in the course directory
            for img_file in course_dir.glob('*.png'):
                print(f"  Testing {img_file.name}...")
                metrics = self._process_image(img_file, expected_course)
                
                if metrics:  # Only append if processing was successful
                    for key in results:
                        results[key].append(metrics[key])
        
        self._generate_report(results)
    
    def _process_image(self, img_path, expected_course):
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to read image: {img_path}")
            return None
        
        # Time the detection
        start_time = time.time()
        course_name, confidence, text_detections = self.detector.detect_course(frame)
        process_time = time.time() - start_time
        
        # Extract flag confidence and course confidence from detector
        flag_detected = course_name is not None
        
        # Draw and save visualization
        vis_frame = self.draw_detection_results(
            frame, 
            flag_detected, 
            confidence, 
            text_detections, 
            course_name
        )
        
        # Save visualization
        output_filename = f"{expected_course}_{img_path.stem}_detected.png"
        output_path = self.output_dir / output_filename
        cv2.imwrite(str(output_path), vis_frame)
        
        return {
            'flag_detection': flag_detected,  # We assume all images should detect a flag
            'ocr_results': self._evaluate_ocr(text_detections, course_name, expected_course),
            'processing_times': process_time,
            'flag_confidences': confidence if flag_detected else 0.0,
            'course_confidences': self._get_best_course_match(course_name)[1] if course_name else 0.0
        }
    
    def _evaluate_ocr(self, text_detections, matched_course, expected_course):
        if not text_detections:
            return {'raw_text': '', 'matched_course': None, 'expected_course': expected_course, 'confidence': 0.0}
        
        best_match, confidence = self._get_best_course_match(matched_course)
        return {
            'raw_text': ', '.join([det['text'] for det in text_detections]),
            'matched_course': matched_course,
            'expected_course': expected_course,
            'confidence': confidence
        }
    
    def _get_best_course_match(self, text):
        """Match text to known course names and return (match, confidence)."""
        if not text:
            return None, 0.0
        
        def normalize_text(s):
            # Remove extra spaces and convert to lowercase
            return ' '.join(s.lower().split())
        
        best_match = None
        best_ratio = 0.0
        
        normalized_text = normalize_text(text)
        for course in self.known_courses:
            normalized_course = normalize_text(course)
            ratio = SequenceMatcher(None, normalized_text, normalized_course).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = course
                
        # If confidence is too low, return no match
        if best_ratio < 0.7:
            return None, 0.0
        
        return best_match, best_ratio
    
    def _generate_report(self, results):
        # Create metrics directory
        Path('metrics').mkdir(exist_ok=True)
        
        with open('metrics/course_detector_report.txt', 'w') as f:
            f.write("Course Detector Evaluation Report\n")
            f.write("===============================\n\n")
            
            # Flag Detection Metrics
            f.write("Flag Detection Metrics:\n")
            f.write("----------------------\n")
            flag_accuracy = np.mean(results['flag_detection'])
            f.write(f"Accuracy: {flag_accuracy:.4f}\n")
            f.write(f"Average Flag Confidence: {np.mean(results['flag_confidences']):.4f}\n\n")
            
            # OCR Metrics
            f.write("OCR Performance Metrics:\n")
            f.write("----------------------\n")
            successful_ocr = [r for r in results['ocr_results'] if r['matched_course'] is not None]
            ocr_success_rate = len(successful_ocr) / len(results['ocr_results'])
            f.write(f"OCR Success Rate: {ocr_success_rate:.4f}\n")
            
            if successful_ocr:
                avg_ocr_confidence = np.mean([r['confidence'] for r in successful_ocr])
                f.write(f"Average OCR Confidence: {avg_ocr_confidence:.4f}\n")
            
            # Timing Metrics
            f.write("\nTiming Metrics:\n")
            f.write("--------------\n")
            avg_time = np.mean(results['processing_times']) * 1000  # Convert to ms
            std_time = np.std(results['processing_times']) * 1000
            fps = 1.0 / np.mean(results['processing_times'])
            f.write(f"Average Processing Time: {avg_time:.2f}ms\n")
            f.write(f"Std Dev Processing Time: {std_time:.2f}ms\n")
            f.write(f"Frames Per Second: {fps:.2f}\n\n")
            
            # OCR Examples
            f.write("OCR Examples:\n")
            f.write("------------\n")
            for i, ocr_result in enumerate(results['ocr_results']):
                if ocr_result['raw_text']:  # Only show examples where text was detected
                    f.write(f"\nExample {i+1}:\n")
                    f.write(f"Raw Text: {ocr_result['raw_text']}\n")
                    f.write(f"Matched Course: {ocr_result['matched_course']}\n")
                    f.write(f"Confidence: {ocr_result['confidence']:.4f}\n")
                    if i >= 9:  # Show only first 10 examples
                        break

def main():
    """Test the detector with course images."""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Test course detector with different flag detection methods')
    parser.add_argument('model_path', 
                       help='Path to model: CNN (.pth file) or template image (.png/.jpg)')
    parser.add_argument('--method', choices=['cnn', 'template'], default='cnn',
                      help='Detection method to use (default: cnn)')
    parser.add_argument('--test_dir', default='Images/Courses',
                      help='Directory containing test images (default: Images/Courses)')
    
    args = parser.parse_args()
    
    # Validate input paths based on method
    if args.method == 'template':
        if not args.model_path.endswith(('.png', '.jpg', '.jpeg')):
            print("Error: Template method requires an image file (.png, .jpg, .jpeg)")
            print("Example: python test_course_detector.py templates/checkered_flag.png --method template")
            return
    else:  # CNN method
        if not args.model_path.endswith('.pth'):
            print("Error: CNN method requires a .pth model file")
            print("Example: python test_course_detector.py models/flag_detector.pth --method cnn")
            return
    
    courses_dir = Path(args.test_dir)
    
    # Initialize tester with selected method
    try:
        tester = CourseDetectorTester(args.model_path, courses_dir, use_template=(args.method == 'template'))
        tester.test_on_dataset()
        print("Testing complete. Results saved in metrics/course_detector_report.txt")
        print("Visualizations saved in metrics/visualizations/")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nMake sure you're using:")
        print("- A .pth model file for CNN method")
        print("- A template image file for template method")

if __name__ == "__main__":
    main()