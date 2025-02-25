import cv2
from development.course_detection.course_detection import CourseDetector
from development.opening_detection.opening_detector import OpeningDetector

def main():
    frame = cv2.imread('development/Images/Courses/WariosGoldMine/WariosGoldMine_57.png')
    frame = cv2.imread('development/Images/RawCourses/MushroomGorge_0.png')
    frame = cv2.imread('development/Images/Courses/Opening/Opening_11.png')
    while True:
        models = dict()
        models['course_detector'] = CourseDetector(flag_model_path='development/course_detection/models/flag_detector_20250221_122742.pth')
        models['opening_detector'] = OpeningDetector()
        #course_name, confidence, text_detections = models['course_detector'].detect_course(frame)
        #print(course_name, confidence, text_detections)
        x = models['opening_detector'].predict(frame)
        print(x)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
     main()
