import cv2
from development.course_detection.course_detection import CourseDetector
from development.opening_detection.opening_detector import OpeningDetector
from development.state_detection.state_detector import StateDetector


def main():
    frame = cv2.imread('development/Images/MenuScreen/characters/characters_0.png')
    while True:
        models = dict()
        models['state_detector'] = StateDetector(model_path = 'production/models/menu_detection.pth')
        screen = models['state_detector'].predict(frame)
        print(screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
     main()
