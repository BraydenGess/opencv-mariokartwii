import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from train_course_detector import SimpleCNN

class CourseFlagDetector:
    def __init__(self, flag_model_path=None, course_model_path=None, device='cpu'):
        self.flag_model_path = flag_model_path
        self.course_model_path = course_model_path
        self.device = device
        self.flag_model = self.load_model(self.flag_model_path, num_classes=2)  # Flag model (2 classes)
        self.course_model = self.load_model(self.course_model_path, num_classes=3)  # Course model (36 classes)

    def load_model(self, model_path, num_classes):
        model = SimpleCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))  # Load the saved weights
        model.eval()  # Set the model to evaluation mode
        return model

    def predict(self, frame):
        class_to_idx_flag = ['no_flag', 'flag']  # Define flag classes
        class_to_idx_course =  ['Luigi Circuit', "N64 DK's Jungle Parkway", 'Opening']

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        regions = {
            'flag': (60, 180, 200, 360),  # Flag detection region
            'course': (925, 1000, 1035, 1790)  # Course detection region
        }

        # Crop regions for flag and course detection
        cropped_flag = frame[regions['flag'][0]:regions['flag'][1], regions['flag'][2]:regions['flag'][3]]
        cropped_course = frame[regions['course'][0]:regions['course'][1], regions['course'][2]:regions['course'][3]]

        # Apply transformations and make predictions for flag
        cropped_flag = transform(cropped_flag).unsqueeze(0).to(self.device)  # Add batch dimension and send to device
        with torch.no_grad():
            flag_output = self.flag_model(cropped_flag)
            flag_probs = F.softmax(flag_output, dim=1)
            flag_confidence, flag_predicted = torch.max(flag_probs, 1)
            flag_class = class_to_idx_flag[flag_predicted.item()]

        # Apply transformations and make predictions for course
        cropped_course = transform(cropped_course).unsqueeze(0).to(self.device)  # Add batch dimension and send to device
        with torch.no_grad():
            course_output = self.course_model(cropped_course)
            course_probs = F.softmax(course_output, dim=1)
            course_confidence, course_predicted = torch.max(course_probs, 1)
            course_class = class_to_idx_course[course_predicted.item()]

        # Return the predictions for both flag and course
        return {
            'flag': {'class': flag_class, 'confidence': flag_confidence.item()},
            'course': {'class': course_class, 'confidence': course_confidence.item()}
        }

model = CourseFlagDetector(flag_model_path='development/secret/flag_detector.pth',
                           course_model_path='development/secret/course_detector.pth')
frame = cv2.imread("development/Images/Courses/N64 DK's Jungle Parkway/N64DKsJungleParkway_110.png")
#frame = cv2.imread("development/Images/Courses/Opening/Opening_15.png")
preds = model.predict(frame = frame)
print(preds)