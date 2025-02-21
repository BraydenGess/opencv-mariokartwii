import sys
from pathlib import Path
import time

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Use absolute import
from course_detection import CourseDetector

import cv2
import argparse
from collections import Counter
import json
from datetime import datetime

class DetectionTester:
    def __init__(self, model_path):
        self.detector = CourseDetector(model_path)
        
    def test_directory(self, test_dir, debug=False, test_psm=False):
        """Test all PNG files in directory and subdirectories"""
        test_path = Path(test_dir)
        if not test_path.exists():
            raise ValueError(f"Test directory not found: {test_dir}")
            
        if test_psm:
            return self._test_psm_modes(test_path, debug)
        else:
            return self._test_normal(test_path, debug)

    def _test_psm_modes(self, test_path, debug):
        """Test all PSM modes (0-14) on the dataset"""
        image_files = list(test_path.glob('**/*.png'))
        if not image_files:
            raise ValueError(f"No PNG files found in {test_path}")
            
        print(f"\nTesting {len(image_files)} images across all PSM modes...")
        
        psm_results = {}
        
        # Test each PSM mode
        for psm in range(14):
            print(f"\nTesting PSM mode {psm}...")
            self.detector.psm = psm
            
            # Initialize timing stats
            flag_times = []
            ocr_times = []
            total_times = []
            
            # Initialize other counters as before
            total_images = len(image_files)
            flag_detections = 0
            course_detections = Counter()
            course_confidences = {}
            
            for img_file in image_files:
                if debug:
                    print(f"\nProcessing: {img_file}")
                
                frame = cv2.imread(str(img_file))
                if frame is None:
                    print(f"Failed to read image: {img_file}")
                    continue
                
                # Time the entire detection process
                start_total = time.time()
                
                # Time flag detection
                start_flag = time.time()
                flag_detected, flag_confidence = self.detector._detect_flag(frame)
                flag_time = time.time() - start_flag
                flag_times.append(flag_time)
                
                if flag_detected:
                    flag_detections += 1
                    # Time OCR process
                    start_ocr = time.time()
                    course_name, confidence, _ = self.detector.detect_course(frame)
                    ocr_time = time.time() - start_ocr
                    ocr_times.append(ocr_time)
                
                total_time = time.time() - start_total
                total_times.append(total_time)
                
                if debug:
                    print(f"PSM {psm} - Course detected: {course_name}")
                    print(f"PSM {psm} - Overall confidence: {confidence}")
                
                # Track course detections and confidences
                if course_name:
                    course_detections[course_name] += 1
                    if course_name not in course_confidences:
                        course_confidences[course_name] = []
                    course_confidences[course_name].append(confidence)
            
            # Add timing stats to results
            psm_results[psm] = {
                'total_images': total_images,
                'flag_detection_rate': flag_detections / total_images,
                'course_detection_rate': sum(course_detections.values()) / total_images,
                'timing_stats': {
                    'flag_detection': {
                        'avg': sum(flag_times) / len(flag_times),
                        'min': min(flag_times),
                        'max': max(flag_times)
                    },
                    'ocr': {
                        'avg': sum(ocr_times) / len(ocr_times) if ocr_times else 0,
                        'min': min(ocr_times) if ocr_times else 0,
                        'max': max(ocr_times) if ocr_times else 0
                    },
                    'total': {
                        'avg': sum(total_times) / len(total_times),
                        'min': min(total_times),
                        'max': max(total_times)
                    }
                },
                'course_detections': {
                    course: {
                        'count': count,
                        'ratio': count / total_images,
                        'avg_confidence': sum(course_confidences[course]) / len(course_confidences[course])
                        if course in course_confidences else 0
                    }
                    for course, count in course_detections.items()
                }
            }
            
            # Print timing summary for this PSM mode
            print(f"\nTiming Stats for PSM {psm}:")
            print(f"Flag Detection: {psm_results[psm]['timing_stats']['flag_detection']['avg']*1000:.1f}ms avg")
            print(f"OCR: {psm_results[psm]['timing_stats']['ocr']['avg']*1000:.1f}ms avg")
            print(f"Total: {psm_results[psm]['timing_stats']['total']['avg']*1000:.1f}ms avg")
        
        # Save comprehensive PSM report
        report_dir = Path('metrics')
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"psm_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Print summary of all PSM modes
        print("\nPSM Mode Comparison:")
        print("=" * 50)
        for psm, results in psm_results.items():
            print(f"\nPSM {psm}:")
            print(f"Course Detection Rate: {results['course_detection_rate']:.2%}")
            print(f"Flag Detection Rate: {results['flag_detection_rate']:.2%}")
            print(f"Unique Courses Detected: {len(results['course_detections'])}")
            print("\nCourse Breakdown:")
            for course, stats in results['course_detections'].items():
                print(f"  {course}: {stats['count']} ({stats['ratio']:.2%})")
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(psm_results, f, indent=4)
        print(f"\nDetailed PSM comparison report saved to {report_path}")
        
        return psm_results

    def _test_normal(self, test_path, debug):
        """Test all PNG files in directory and subdirectories"""
        test_path = Path(test_path)
        if not test_path.exists():
            raise ValueError(f"Test directory not found: {test_path}")
            
        # Collect all PNG files
        image_files = list(test_path.glob('**/*.png'))
        if not image_files:
            raise ValueError(f"No PNG files found in {test_path}")
            
        print(f"\nTesting {len(image_files)} images...")
        
        # Initialize counters
        total_images = len(image_files)
        flag_detections = 0
        course_detections = Counter()
        course_confidences = {}
        all_detections = []  # New: Track all detections
        
        # Initialize timing lists
        flag_times = []
        ocr_times = []
        total_times = []
        
        # Process each image
        for img_file in image_files:
            if debug:
                print(f"\nProcessing: {img_file}")
            
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"Failed to read image: {img_file}")
                continue
            
            # Time the entire detection process
            start_total = time.time()
            
            # Time flag detection
            start_flag = time.time()
            flag_detected, flag_confidence = self.detector._detect_flag(frame)
            flag_time = time.time() - start_flag
            flag_times.append(flag_time)
            
            if flag_detected:
                flag_detections += 1
                # Time OCR process
                start_ocr = time.time()
                course_name, confidence, _ = self.detector.detect_course(frame)
                ocr_time = time.time() - start_ocr
                ocr_times.append(ocr_time)
            
            total_time = time.time() - start_total
            total_times.append(total_time)
            
            if debug:
                print(f"Flag detected: {flag_detected}")
                print(f"Flag confidence: {flag_confidence}")
                print(f"Course detected: {course_name}")
                print(f"Overall confidence: {confidence}")
            
            # Track course detections and confidences
            if course_name:
                course_detections[course_name] += 1
                if course_name not in course_confidences:
                    course_confidences[course_name] = []
                course_confidences[course_name].append(confidence)
            
            # New: Log detection details for this image
            detection_entry = {
                'image_path': str(img_file),
                'flag_detected': flag_detected,
                'flag_confidence': flag_confidence,
                'course_detected': course_name if course_name else None,
                'course_confidence': confidence,
                'raw_text': self.detector.last_raw_text if hasattr(self.detector, 'last_raw_text') else None,
                'timing': {
                    'flag_detection': flag_time,
                    'ocr': ocr_time if flag_detected else None,
                    'total': total_time
                }
            }
            all_detections.append(detection_entry)
        
        # Generate report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': total_images,
            'flag_detection_rate': flag_detections / total_images,
            'course_detection_rate': sum(course_detections.values()) / total_images,
            'course_detections': {
                course: {
                    'count': count,
                    'ratio': count / total_images,
                    'avg_confidence': sum(course_confidences[course]) / len(course_confidences[course])
                    if course in course_confidences else 0
                }
                for course, count in course_detections.items()
            },
            'all_detections': all_detections,  # New: Include all detection details
            'timing_stats': {
                'flag_detection': {
                    'avg': sum(flag_times) / len(flag_times),
                    'min': min(flag_times),
                    'max': max(flag_times)
                },
                'ocr': {
                    'avg': sum(ocr_times) / len(ocr_times) if ocr_times else 0,
                    'min': min(ocr_times) if ocr_times else 0,
                    'max': max(ocr_times) if ocr_times else 0
                },
                'total': {
                    'avg': sum(total_times) / len(total_times),
                    'min': min(total_times),
                    'max': max(total_times)
                }
            }
        }
        
        # Print report
        print("\nDetection Results:")
        print("=" * 50)
        print(f"Total Images: {total_images}")
        print(f"Flag Detection Rate: {report['flag_detection_rate']:.2%}")
        print(f"Course Detection Rate: {report['course_detection_rate']:.2%}")
        print("\nCourse-wise Detections:")
        print("-" * 50)
        for course, stats in report['course_detections'].items():
            print(f"{course}:")
            print(f"  Count: {stats['count']}")
            print(f"  Ratio: {stats['ratio']:.2%}")
            print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        
        # Print timing stats in report
        print("\nTiming Statistics:")
        print("-" * 50)
        print(f"Flag Detection: {report['timing_stats']['flag_detection']['avg']*1000:.1f}ms avg")
        print(f"OCR: {report['timing_stats']['ocr']['avg']*1000:.1f}ms avg")
        print(f"Total: {report['timing_stats']['total']['avg']*1000:.1f}ms avg")
        
        # Save report to file
        report_dir = Path('metrics')
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\nDetailed report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Test course detection on directory of images')
    parser.add_argument('model_path', 
                       help='Path to CNN model (.pth file)')
    parser.add_argument('test_dir',
                       help='Directory containing test images')
    parser.add_argument('--test-psm', action='store_true',
                       help='Test all PSM modes')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Validate model path
    if not args.model_path.endswith('.pth'):
        print("Error: Model path must be a .pth file")
        return
    
    try:
        tester = DetectionTester(args.model_path)
        tester.test_directory(args.test_dir, debug=args.debug, test_psm=args.test_psm)
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
