from PIL import Image
from torchvision import transforms
import torch

class StateDetector():
    def __init__(self, model = None):
        self.model = model

    def predict(self,frame):
        # Define the same transformation pipeline as during training
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((100, 20, 800, 160))),  # Crop (left, top, right, bottom)
            transforms.Resize((128, 128)),  # Resize to the same size used during training
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        # Load and transform the image
        image_path = "path_to_image.png"  # Specify the image path
        img = Image.open(image_path).convert('RGB')  # Open the image
        img = transform(img).unsqueeze(0)  # Apply the transformation and add batch dimension

        # Ensure the model is on the correct device (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Send the image tensor to the same device (GPU/CPU)
        img = img.to(device)

        # Make the prediction
        with torch.no_grad():  # No need to track gradients during inference
            output = self.model(img)  # Forward pass through the model

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)

        # Map the predicted class index back to the class name
        predicted_class_name = self.model.classes[predicted_class.item()]

        return predicted_class_name