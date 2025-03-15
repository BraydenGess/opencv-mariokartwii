import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define the regions for flag and text detection
regions = {
    'flag': (60, 180, 200, 360),  # Flag detection region
    'course': (925, 1000, 1035, 1790)  # course detection region
}

def select_device() -> torch.device:
    """
    Selects the best available computing device for PyTorch

    Returns: torch.device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")   # macOS Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPUs
    return torch.device("cpu")

# Custom dataset for loading and transforming images
class RegionDataset(Dataset):
    def __init__(self, root_folder, transform=None, target='flag'):
        self.root_folder = root_folder
        self.transform = transform
        self.target = target  # Either 'flag' or 'text'
        self.image_paths = []
        self.labels = []

        for course_class in os.listdir(root_folder):
            class_path = os.path.join(root_folder, course_class)
            if os.path.isdir(class_path):
                images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
                self.image_paths.extend(images)
                if target == 'flag':
                    self.labels.extend([1 if course_class != "Opening" else 0] * len(images))
                else:
                    self.labels.extend([course_class] * len(images))

        if target == 'course':
            self.classes = sorted(set(self.labels))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.labels = [self.class_to_idx[label] for label in self.labels]
        elif target == 'flag':
            self.classes = ['no_flag', 'flag']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")  # Print the path of the image that failed to load
            return None  # Return None if the image is not loaded

        y1, y2, x1, x2 = regions[self.target]
        cropped_image = image[y1:y2, x1:x2]
        if self.transform:
            cropped_image = self.transform(cropped_image)
        return cropped_image, label

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Function to create data loader
def create_dataloader(root_folder, batch_size=8, target='flag'):
    dataset = RegionDataset(root_folder, transform=transform, target=target)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, dataset.classes

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train function
def train_model(root_folder, target, model_path, num_epochs=5, batch_size=8):
    train_loader, classes = create_dataloader(root_folder, batch_size, target)
    num_classes = len(classes)
    print(f"Training {target} model with classes: {classes}")
    device = select_device()  # Select the device
    model = SimpleCNN(num_classes).to(device)  # Move model to the selected device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move tensors to the selected device
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"{target} Model - Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), model_path)
    print(f"{target.capitalize()} Model saved!")

# Main function
def main():
    root_folder = "development/Images/Courses"
    train_model(root_folder, 'course', 'development/secret/course_detector.pth')
    train_model(root_folder, 'flag', 'development/secret/flag_detector.pth')

if __name__ == "__main__":
    main()

