import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from development.state_detection.train_state_detector import SimpleCNN

class StateDetector:
    def __init__(self, model_path=None, class_names=None):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names =  ['characters', 'controller', 'drift', 'homescreen', 'main',
                             'players', 'startrace', 'vehicles', 'vs', 'None']
        self.model = None
        self.setup()

        # Define transformations (fixing the missing self.transform)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((100, 20, 800, 160))),  # Crop
            transforms.Resize((128, 128)),  # Resize
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

    def setup(self):
        self.model = SimpleCNN(num_classes=10)  # Ensure num_classes matches training
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")

    def predict(self, frame):
        """Predicts the class of a given image frame."""
        # Convert NumPy array (if needed) to PIL image
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)

        # Apply transformations
        img = self.transform(frame).unsqueeze(0).to(self.device)  # Add batch dim & move to device

        # Make the prediction
        with torch.no_grad():
            output = self.model(img)

        # Get the predicted class index
        _, predicted_class = torch.max(output, 1)

        # Map to class name if available
        if self.class_names:
            return self.class_names[predicted_class.item()]

        return predicted_class.item()  # Return class index if no names are provided
