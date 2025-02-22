import random
import cv2


def pause_toggle(frame):
    #pause = predict(frame)
    # if pause:
        #pause
    # if not pause:
        # don't pause
    pass

def play_music(frame, model_store, sp):
    course_name, confidence, text_detections = model_store.models['course_detector'].detect_course(frame)
    if course_name:
        if course_name != sp.course_queued:
            sp.queue_newsong(course_name)
            num = random.randint(1,100)
            cv2.imwrite(f'development/triggers/{course_name}_{num}.png', frame)

    return 0

def run_audio(frame, model_store, sp):
    pause_toggle(frame)
    play_music(frame, model_store, sp)
