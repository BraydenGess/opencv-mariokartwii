import cv2
import numpy as np

from main import select_device
from __init__ import *

# Open the video capture (adjust source as needed)
cap = cv2.VideoCapture(0)  # Change this to the correct HDMI source if necessary

# Get frame width and height
ret, frame = cap.read()
if not ret:
    print("Failed to capture video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

height, width, _ = frame.shape
black_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Create a black frame

block_tv_output = False  # Flag to control HDMI output

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if block_tv_output:
        output_frame = black_frame.copy()  # Send a black screen to HDMI
    else:
        output_frame = frame.copy()

    # Display output
    cv2.imshow("Computer Output", frame)  # Always show the real frame on the computer
    cv2.imshow("TV Output", output_frame)  # Simulated TV output (would need an actual HDMI tool)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):  # Press 'b' to block TV output
        block_tv_output = True
        print("TV output blocked.")
    elif key == ord('r'):  # Press 'r' to resume TV output
        block_tv_output = False
        print("TV output resumed.")
    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

def blackout_mkwii():
    """
    Main function to initialize and run the application
    """
    device = select_device()
    model_store = ModelStore(device=device)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()


if __name__ == "__main__":
     blackout_mkwii()