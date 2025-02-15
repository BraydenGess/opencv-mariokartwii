# python validate.py <model_path>

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from train import MarioKartDataset
from torch.utils.data import DataLoader
import numpy as np
import sys
import time
from statistics import mean

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    batch_times = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Time the batch inference
            batch_start = time.time()
            outputs = model(inputs)
            batch_end = time.time()
            
            # Calculate timing metrics
            batch_time = batch_end - batch_start
            per_image_time = batch_time / batch_size
            
            batch_times.append(batch_time)
            inference_times.extend([per_image_time] * batch_size)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate timing statistics
    avg_batch_time = mean(batch_times)
    avg_inference_time = mean(inference_times)
    total_time = sum(batch_times)
    
    print("\nInference Speed Metrics:")
    print(f"Average batch inference time: {avg_batch_time*1000:.2f}ms")
    print(f"Average per-image inference time: {avg_inference_time*1000:.2f}ms")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Images per second: {len(inference_times)/total_time:.2f}")
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    model_path = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = MarioKartDataset('Images', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Evaluate model
    predictions, labels = evaluate_model(model, test_loader, device)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, 
                              target_names=test_dataset.classes))

    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, test_dataset.classes)

if __name__ == "__main__":
    main()