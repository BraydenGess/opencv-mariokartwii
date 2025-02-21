import cv2
import sys
import time
from development.course_detector import CourseDetector

from spotify_audio import run_audio

def test():
    #cap = cv2.VideoCapture(0)
    detector = CourseDetector(flag_model_path='development/models/flag_detector_20250215_115232.pth')
    #while cap.isOpened():
    while True:
        # ret, frame = cap.read()
        # Draw ROIs for debugging
        frame = cv2.imread('development/Images/Courses/N64DKsJungleParkway/N64DKsJungleParkway_54.png')
        x, y, w, h = detector.text_roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Time the detection
        start_time = time.time()
        course_name, confidence, text_detections = detector.detect_course(frame)
        print(course_name, confidence, text_detections)
        process_time = time.time() - start_time

        # Display results
        if course_name:
            cv2.putText(frame, f"{course_name} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {1 / process_time:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cap.release()
    cv2.destroyAllWindows()

def main():
    frame = cv2.imread('development/Images/Courses/N64DKsJungleParkway/N64DKsJungleParkway_54.png')
    #cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    while True:
        run_audio(frame)
        return 0



if __name__ == "__main__":
     main()

