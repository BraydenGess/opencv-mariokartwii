import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


# Define the coordinates for the regions
regions = {
    'region1': (432, 512, 305, 855),  # Top-left region
    'region2': (432, 512, 1045, 1595),  # Top-right region
    'region3': (797, 877, 305, 855),  # Bottom-left region
    'region4': (797, 877, 1045, 1595)  # Bottom-right region
}


# Custom dataset for loading and transforming images
class RegionDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

        # Auto-fill the classes by extracting unique labels from filenames
        self.classes = self._get_classes_from_filenames()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _get_classes_from_filenames(self):
        # Set to hold unique classes extracted from filenames
        unique_classes = set()

        for base_path in self.image_paths:
            filename = os.path.basename(base_path)
            class_labels = filename.split('_')[0].split('+')
            unique_classes.update(class_labels)

        return sorted(list(unique_classes))  # Return sorted list of unique class labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)

        # Split the filename to get class labels
        class_labels = filename.split('_')[0].split('+')  # e.g., ['A', 'B', 'C', 'D']

        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Extract the regions from the image
        cropped_regions = {region_name: image[y1:y2, x1:x2] for region_name, (y1, y2, x1, x2) in regions.items()}

        # Prepare the labeled regions
        labeled_regions = []
        for i, (region_name, cropped_image) in enumerate(cropped_regions.items()):
            class_label = class_labels[i]  # The class corresponding to this region
            class_idx = self.class_to_idx[class_label]  # Get the index for the class
            if self.transform:
                cropped_image = self.transform(cropped_image)
            labeled_regions.append((cropped_image, class_idx))

        return labeled_regions


# Define transformations (resize, convert to tensor)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL image
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
])


# Function to create the data loader
def create_dataloader(image_folder, batch_size=8):
    dataset = RegionDataset(image_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, dataset.classes  # Return the data loader and the list of classes


# Simple CNN Model for Region Classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjusted for 128x128 input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to preprocess the image and make predictions
def predict(image_path, model, class_to_idx):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Define the regions for cropping (same as during training)
    regions = {
        'region1': (432, 512, 305, 855),  # Top-left region
        'region2': (432, 512, 1045, 1595),  # Top-right region
        'region3': (797, 877, 305, 855),  # Bottom-left region
        'region4': (797, 877, 1045, 1595)  # Bottom-right region
    }

    # Extract the regions from the image
    cropped_regions = {region_name: image[y1:y2, x1:x2] for region_name, (y1, y2, x1, x2) in regions.items()}

    # Prepare the labeled regions
    region_predictions = []
    for region_name, cropped_image in cropped_regions.items():
        # Apply transformations (resize, convert to tensor)
        cropped_image = transform(cropped_image)
        cropped_image = cropped_image.unsqueeze(0)  # Add batch dimension (1, C, H, W)

        # Make prediction
        with torch.no_grad():
            outputs = model(cropped_image)  # Pass the image through the model
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
            predicted_class = predicted.item()  # Convert tensor to scalar

            # Get the class label from the index
            class_label = list(class_to_idx.keys())[predicted_class]
            region_predictions.append((region_name, class_label))

    return region_predictions

# Main function for training and evaluating the model
def main():
    image_folder = "development/Images/Vehicles"  # Path to your folder with images
    batch_size = 8
    num_epochs = 15

    # Create the data loaders and get class labels
    train_loader, classes = create_dataloader(image_folder, batch_size=batch_size)

    # Get the number of classes (auto-filled from dataset)
    num_classes = len(classes)
    print(f"Classes: {classes}")

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            for images, labels in batch:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training complete!")

    # Save the model
    torch.save(model.state_dict(), 'production/models/vehicle_classifier.pth')
    print(classes)
    print("Model saved!")


# Run the main function
if __name__ == "__main__":
    main()