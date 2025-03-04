from development.placement_detection.train_placement_detection import SimpleCNN
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL image
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),  # Resize images to match model input
    transforms.ToTensor(),  # Convert to tensor
])

class PlacementDetector:
    def __init__(self, model_path = None, device = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        model = SimpleCNN(num_classes = 12)
        model.load_state_dict(torch.load(self.model_path))  # Load the saved weights
        model.eval()  # Set the model to evaluation mode
        return model

    # Function to preprocess the image and make predictions
    def predict(self, frame):
        classes = ['1', '10', '11', '12', '2', '3', '4', '5', '6', '7', '8', '9']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        # Read the image using OpenCV

        # Define the regions for cropping (same as during training)
        regions = {
            'region1': (370, 510, 100, 320),  # Top-left region
            'region2': (370, 510, 1580, 1800),  # Top-right region
            'region3': (880, 1020, 100, 320),  # Bottom-left region
            'region4': (880, 1020, 1580, 1800)  # Bottom-right region
        }

        # Extract the regions from the image
        cropped_regions = {region_name: frame[y1:y2, x1:x2] for region_name, (y1, y2, x1, x2) in regions.items()}

        # Prepare the labeled regions
        region_predictions = []
        for region_name, cropped_image in cropped_regions.items():
            # Apply transformations (resize, convert to tensor)
            cropped_image = transform(cropped_image)
            cropped_image = cropped_image.unsqueeze(0)  # Add batch dimension (1, C, H, W)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(cropped_image)  # Pass the image through the model
                probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                confidence, predicted = torch.max(probabilities, 1)  # Get the highest probability and its index
                predicted_class = predicted.item()  # Convert tensor to scalar

                # Get the class label from the index
                class_label = list(class_to_idx.keys())[predicted_class]
                region_predictions.append((region_name, class_label, confidence.item()))  # Store confidence as well

        return region_predictions
