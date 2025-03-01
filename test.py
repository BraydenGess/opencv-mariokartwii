import cv2
from development.character_detection.character_detector import CharacterDetector

frame = cv2.imread(filename ='development/Images/Vehicles/Flame Runner+Flame Runner+Flame Runner+Flame Runner_0.png')

frame1 = frame[432:512,305:855]
frame2 = frame[432:512,1045:1595]
frame3 = frame[797:877,305:855]
frame4 = frame[797:877,1045:1595]

if frame is None:
    print("Error: Could not load image.")
else:
    # Display the image
    cv2.imshow("Frame", frame1)
    cv2.imshow("Frame2", frame2)
    cv2.imshow("Frame3", frame3)
    cv2.imshow("Frame4", frame4)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()