import cv2
from development.character_detection.character_detector import CharacterDetector

frame = cv2.imread(filename ='development/Images/Placement/12+11+1+10_MoonviewHighway_411.png')
frame9 = cv2.imread(filename ='development/Images/Placement/12+11+10+1_GrumbleVolcano_95.png')
frame10 = cv2.imread(filename ='development/Images/Placement/12+11+10+1_GrumbleVolcano_95.png')


frame1 = frame[370:510,100:320]
frame2 = frame10[370:510,1580:1800]
frame3 = frame9[880:1020,100:320]
frame4 = frame[880:1020,1580:1800]

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