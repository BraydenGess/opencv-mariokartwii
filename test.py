import cv2
import torch
import time
from development.course_detection.course_detection import CourseDetector
from development.state_detection.state_detector import StateDetector
import threading
import queue

def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # macOS Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPUs
    else:
        return torch.device("cpu")  # Default to CPU

def update_frames(cap, frame_queue):
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(new_frame)
    cap.release()

def detect_course(models, frame_queue):
    while True:
        frame = frame_queue.get()  # Wait until a frame is available
        t1 = time.time()
        course_name, confidence, text_detections = models['course_detector'].detect_course(frame)
        t2 = time.time()
        print(f"Course Detection: {course_name} | Time Taken: {t2 - t1:.4f} sec")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_state(models, frame_queue):
    while True:
        frame = frame_queue.get()  # Wait until a frame is available
        t1 = time.time()
        screen, confidence = models['state_detector'].predict(frame)
        t2 = time.time()
        print(f"State Detection: {screen} | Time Taken: {t2 - t1:.4f} sec")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    device = select_device()
    models = {
        'course_detector': CourseDetector(flag_model_path='development/course_detection/models/flag_detector_20250221_122742.pth',
                                          device=device),
        'state_detector': StateDetector(model_path='production/models/menu_detection.pth',
                                        device=device)
    }
    # Initialize frame queues
    frame_queue = queue.Queue(maxsize=1)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use a file path if processing a video

    # Start threads
    frame_thread = threading.Thread(target=update_frames, args=(cap, frame_queue), daemon=True)
    course_thread = threading.Thread(target=detect_course, args=(models, frame_queue), daemon=True)
    state_thread = threading.Thread(target=detect_state, args=(models, frame_queue), daemon=True)

    frame_thread.start()
    course_thread.start()
    state_thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
