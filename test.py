import cv2
from development.character_detection.character_detector import CharacterDetector

frame = cv2.imread(filename = 'development/Images/Characters/KoopaTroopa+DryBones+Mario+Luigi_0.png')
frame2 = cv2.imread(filename = 'development/Images/Characters/Toadette+KoopaTroopa+DryBones+Mario_0.png')
frame3 = cv2.imread(filename = 'development/Images/Characters/Toad+Toadette+KoopaTroopa+DryBones_0.png')

cd = CharacterDetector(model_path = 'production/models/character_classifier.pth')
p = cd.predict(frame2)
print(p)
exit()

frame = frame[445:485,120:420]
frame2 = frame2[445:485,1485:1785]
frame3 = frame3[845:885,120:420]
frame4 = frame3[845:885,1485:1785]

if frame is None:
    print("Error: Could not load image.")
else:
    # Display the image
    cv2.imshow("Frame", frame)
    #cv2.imshow("Frame2", frame2)
    cv2.imshow("Frame3", frame3)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()