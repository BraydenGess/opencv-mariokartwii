import torch
import torch.nn as nn
import torch.nn.functional as F  # Import softmax function
from torchvision import transforms
from PIL import Image
from development.countdown_detection.train_countdown_detector  import SimpleCNN
import cv2

class CountdownDetector:
    def __init__(self, model_path=None, class_names=None, device = None):
        self.model_path = model_path
        self.device = device
        self.class_names =  ['1', '2', '3', 'FINISH', 'GO']
        self.model = None
        self.setup()

        # Define transformations (fixing the missing self.transform)
        self.transform = transforms.Compose([
            #transforms.Lambda(lambda img: img.crop((100, 190, 850, 340))),  # Crop
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),  # Resize
            transforms.ToTensor(),  # Convert to tensor
        ])

    def setup(self):
        self.model = SimpleCNN(num_classes=5)  # Ensure num_classes matches training
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode-
        #print("Model loaded successfully!")

    def predict(self, frame):
        """Predicts the class of a given image frame with confidence score."""
        regions = [
            ['region 1', 190, 340, 100, 850],
            ['region 2', 190, 340, 1035, 1785],
            ['region 3', 730, 880, 100, 850],
            ['region 4', 730, 880, 1035, 1785],
        ]
        labels = []
        for i in range(4):
            region = regions[i][0]
            y0, y1, x0, x1 = regions[i][1], regions[i][2], regions[i][3], regions[i][4]
            new_frame = frame[y0:y1, x0:x1]
            if isinstance(new_frame, torch.Tensor):
                new_frame = new_frame.cpu().numpy()
            if not isinstance(new_frame, Image.Image):
                new_frame = Image.fromarray(new_frame)
            img = self.transform(new_frame).unsqueeze(0).to(self.device)  # Add batch dim & move to device
            with torch.no_grad():
                output = self.model(img)  # Raw model outputs (logits)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)  # Get max probability & class index
            if self.class_names:
                labels.append([region,self.class_names[predicted_class.item()], confidence.item()])
        return labels