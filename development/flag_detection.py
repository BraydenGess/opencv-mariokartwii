# python flag_detection.py <dataset_path>

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
import sys
import cv2

class FlagDataset(Dataset):
    def __init__(self, root_dir, transform=None, visualize_roi=False):
        self.root_dir = root_dir
        self.transform = transform
        self.visualize_roi = visualize_roi
        self.classes = ['no_flag', 'flag']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        # Define ROI for flag detection
        self.flag_roi = (250, 75, 150, 150)  # (x, y, width, height)
        
        # Create visualization directory if needed
        if self.visualize_roi:
            self.vis_dir = Path('roi_visualizations')
            self.vis_dir.mkdir(exist_ok=True)
        
        # Load all images and labels
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])

    def visualize_roi_on_image(self, image_path, roi_image):
        """Save original image with ROI overlay"""
        # Load original image with cv2
        orig_image = cv2.imread(image_path)
        x, y, w, h = self.flag_roi
        
        # Draw ROI rectangle in red
        cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Add semi-transparent overlay
        overlay = orig_image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, orig_image, 0.7, 0, orig_image)
        
        # Save visualization
        vis_path = self.vis_dir / f"roi_{Path(image_path).name}"
        cv2.imwrite(str(vis_path), orig_image)
        
        # Also save the cropped ROI next to it for comparison
        roi_vis_path = self.vis_dir / f"roi_crop_{Path(image_path).name}"
        cv2.imwrite(str(roi_vis_path), cv2.cvtColor(np.array(roi_image), cv2.COLOR_RGB2BGR))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # Extract ROI
        x, y, w, h = self.flag_roi
        roi_image = image.crop((x, y, x+w, y+h))

        # Visualize if requested
        if self.visualize_roi:
            self.visualize_roi_on_image(img_path, roi_image)

        if self.transform:
            roi_image = self.transform(roi_image)

        return roi_image, label

class FlagDetector(nn.Module):
    def __init__(self):
        super(FlagDetector, self).__init__()
        # Use a small ResNet for faster inference
        self.model = models.resnet18(weights=None)
        # Modify the first conv layer to take a smaller input
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Track best model state
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Save only the best model at the end of training
    if best_model_state is not None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'models/flag_detector_{timestamp}.pth'
        torch.save(best_model_state, model_path)
        print(f'Best model saved with validation accuracy: {best_acc:.4f}')

    return model

def main():
    # Set up dataset with visualization enabled
    dataset_path = sys.argv[1]
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets with visualization enabled for training set
    dataset = FlagDataset(dataset_path, transform=None, visualize_roi=True)
    
    # Process all images to generate visualizations
    print("Generating ROI visualizations...")
    for i in range(len(dataset)):
        dataset[i]
    print(f"ROI visualizations saved in {dataset.vis_dir}/")
    
    # Now create the actual training dataset with transforms
    dataset = FlagDataset(dataset_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = FlagDetector().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()