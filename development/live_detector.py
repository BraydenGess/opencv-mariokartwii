import cv2
import argparse
from pathlib import Path
from course_detector import CourseDetector
import time

def live_test(model_path, use_template=False, camera_id=0):
    """Run live testing with webcam feed, printing course detections"""
    detector = CourseDetector(model_path, use_template=use_template)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    print("\nLive Detection Started")
    print("Press 'q' to quit")
    print(f"Using {'Template' if use_template else 'CNN'} detection method")
    print("-" * 50)
    
    last_detection = None
    last_detection_time = 0
    cooldown = 2  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        course_name, confidence, _ = detector.detect_course(frame)
        
        current_time = time.time()
        if course_name:
            if (course_name != last_detection or 
                current_time - last_detection_time > cooldown):
                print(f"Course Detected: {course_name} (confidence: {confidence:.2f})")
                last_detection = course_name
                last_detection_time = current_time
        
        cv2.imshow('Live Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Live course detection testing')
    parser.add_argument('model_path', 
                       help='Path to model: CNN (.pth file) or template image (.png/.jpg)')
    parser.add_argument('--method', choices=['cnn', 'template'], default='cnn',
                      help='Detection method to use (default: cnn)')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    if args.method == 'template':
        if not args.model_path.endswith(('.png', '.jpg', '.jpeg')):
            print("Error: Template method requires an image file (.png, .jpg, .jpeg)")
            print("Example: python live_detector.py templates/checkered_flag.png --method template")
            return
    else:
        if not args.model_path.endswith('.pth'):
            print("Error: CNN method requires a .pth model file")
            print("Example: python live_detector.py models/flag_detector.pth --method cnn")
            return
    
    try:
        live_test(
            args.model_path, 
            use_template=(args.method == 'template'),
            camera_id=args.camera
        )
    except Exception as e:
        print(f"Error during live testing: {e}")

if __name__ == "__main__":
    main()