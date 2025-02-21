import subprocess
from pathlib import Path
import json
from datetime import datetime
import statistics

def analyze_courses(model_path, courses_dir):
    courses_path = Path(courses_dir)
    if not courses_path.exists():
        raise ValueError(f"Courses directory not found: {courses_dir}")
    
    # Store results for each course
    all_results = {}
    
    # Process each course directory
    for course_dir in sorted(courses_path.glob("*/")):
        if not course_dir.is_dir():
            continue
            
        course_name = course_dir.name
        print(f"\nProcessing course: {course_name}")
        
        # Run detection test with PSM analysis
        cmd = ["python", "test_detection.py", model_path, str(course_dir), "--test-psm"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse the JSON report (most recent file in metrics directory)
        metrics_dir = Path("metrics")
        report_files = list(metrics_dir.glob("psm_detection_report_*.json"))
        if not report_files:
            print(f"No report found for {course_name}")
            continue
            
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        with open(latest_report) as f:
            course_results = json.load(f)
            all_results[course_name] = course_results

    # Generate comprehensive report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': analyze_summary(all_results),
        'course_details': analyze_course_details(all_results)
    }
    
    # Save report
    report_path = Path('metrics') / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print report
    print_report(report)
    
def analyze_summary(all_results):
    """Generate summary statistics across all courses"""
    best_psm_counts = {}
    total_detection_rates = []
    total_flag_rates = []
    
    for course_results in all_results.values():
        # Find best PSM for this course
        best_psm = max(course_results.keys(), 
                      key=lambda p: course_results[p]['course_detection_rate'])
        best_psm_counts[best_psm] = best_psm_counts.get(best_psm, 0) + 1
        
        # Collect rates for averaging
        best_results = course_results[best_psm]
        total_detection_rates.append(best_results['course_detection_rate'])
        total_flag_rates.append(best_results['flag_detection_rate'])
    
    return {
        'total_courses_analyzed': len(all_results),
        'average_detection_rate': statistics.mean(total_detection_rates),
        'average_flag_rate': statistics.mean(total_flag_rates),
        'best_psm_distribution': best_psm_counts
    }

def analyze_course_details(all_results):
    """Generate per-course detailed statistics"""
    course_details = {}
    
    for course_name, course_results in all_results.items():
        # Find best PSM for this course
        best_psm = max(course_results.keys(), 
                      key=lambda p: course_results[p]['course_detection_rate'])
        best_results = course_results[best_psm]
        
        course_details[course_name] = {
            'best_psm': best_psm,
            'detection_rate': best_results['course_detection_rate'],
            'flag_rate': best_results['flag_detection_rate'],
            'total_images': best_results['total_images']
        }
    
    return course_details

def print_report(report):
    """Print formatted report to console"""
    print("\n" + "="*50)
    print("COMPREHENSIVE DETECTION ANALYSIS")
    print("="*50)
    
    # Summary
    summary = report['summary']
    print("\nOVERALL SUMMARY:")
    print(f"Total Courses Analyzed: {summary['total_courses_analyzed']}")
    print(f"Average Detection Rate: {summary['average_detection_rate']:.2%}")
    print(f"Average Flag Detection Rate: {summary['average_flag_rate']:.2%}")
    
    print("\nBest PSM Distribution:")
    for psm, count in summary['best_psm_distribution'].items():
        print(f"PSM {psm}: {count} courses")
    
    # Course Details
    print("\nPER-COURSE DETAILS:")
    print("-"*50)
    for course, details in report['course_details'].items():
        print(f"\n{course}:")
        print(f"  Best PSM: {details['best_psm']}")
        print(f"  Detection Rate: {details['detection_rate']:.2%}")
        print(f"  Flag Detection Rate: {details['flag_rate']:.2%}")
        print(f"  Total Images: {details['total_images']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze course detection across all courses')
    parser.add_argument('model_path', help='Path to CNN model (.pth file)')
    parser.add_argument('courses_dir', help='Directory containing course subdirectories')
    
    args = parser.parse_args()
    
    analyze_courses(args.model_path, args.courses_dir)