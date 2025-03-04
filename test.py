import cv2
from development.countdown_detection.countdown_detector import CountdownDetector

frame = cv2.imread(filename ='development/Images/Countdown/FINISH/FINISH_LuigiCircuit_328.png')
frame9 = cv2.imread(filename ='development/Images/Placement/12+11+10+1_GrumbleVolcano_95.png')
frame10 = cv2.imread(filename ='development/Images/Placement/12+11+10+1_GrumbleVolcano_95.png')

model = CountdownDetector(model_path = 'production/models/countdown_detection.pth',device = 'mps')
print(model.predict(frame))
exit()

frame1 = frame[190:340,100:850]
frame2 = frame10[370:510,1580:1800]
frame3 = frame9[880:1020,100:320]
frame4 = frame[880:1020,1580:1800]

if frame is None:
    print("Error: Could not load image.")
else:
    # Display the image
    cv2.imshow("Frame", frame1)
    #cv2.imshow("Frame2", frame2)
    #cv2.imshow("Frame3", frame3)
    #cv2.imshow("Frame4", frame4)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()