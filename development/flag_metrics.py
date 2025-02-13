# python flag_metrics.py <model_path> <test_data_path>

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import sys
from flag_detection import FlagDataset, FlagDetector

class FlagMetricsEvaluator:
    def __init__(self, model_path, test_data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = FlagDetector().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Setup data
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.test_dataset = FlagDataset(test_data_path, transform=self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        all_preds = []
        all_labels = []
        all_probs = []
        batch_times = []
        inference_times = []
        
        print("\nStarting evaluation...")
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Time the batch inference
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_start = time.time()
                
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_end = time.time()
                
                # Calculate timing metrics
                batch_time = batch_end - batch_start
                per_image_time = batch_time / batch_size
                
                batch_times.append(batch_time)
                inference_times.extend([per_image_time] * batch_size)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of flag class
        
        return (np.array(all_preds), np.array(all_labels), np.array(all_probs), 
                batch_times, inference_times)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Flag', 'Flag'],
                   yticklabels=['No Flag', 'Flag'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('metrics/confusion_matrix.png')
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_prob):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('metrics/roc_curve.png')
        plt.close()
        
        return roc_auc
    
    def analyze_timing(self, batch_times, inference_times):
        """Analyze and return timing metrics"""
        timing_stats = {
            'avg_batch_time': np.mean(batch_times) * 1000,  # Convert to ms
            'std_batch_time': np.std(batch_times) * 1000,
            'avg_inference_time': np.mean(inference_times) * 1000,
            'std_inference_time': np.std(inference_times) * 1000,
            'fps': 1.0 / np.mean(inference_times),
            'total_images': len(inference_times),
            'total_time': sum(batch_times)
        }
        return timing_stats
    
    def run_evaluation(self):
        """Run complete evaluation and save results"""
        # Create metrics directory if it doesn't exist
        Path('metrics').mkdir(exist_ok=True)
        
        # Run evaluation
        preds, labels, probs, batch_times, inference_times = self.evaluate_model()
        
        # Calculate all metrics
        cm = self.plot_confusion_matrix(labels, preds)
        roc_auc = self.plot_roc_curve(labels, probs)
        timing_stats = self.analyze_timing(batch_times, inference_times)
        
        # Get classification report with consistent class names
        class_report = classification_report(labels, preds, 
                                          target_names=['no_flag', 'flag'],
                                          output_dict=True)
        
        # Save detailed results
        with open('metrics/evaluation_report.txt', 'w') as f:
            f.write("Flag Detection Model Evaluation Report\n")
            f.write("=====================================\n\n")
            
            f.write("Classification Metrics:\n")
            f.write("-----------------------\n")
            f.write(f"Accuracy: {class_report['accuracy']:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write("\nPer-Class Metrics:\n")
            
            # Use consistent class names
            display_names = {'no_flag': 'No Flag', 'flag': 'Flag'}
            for cls in ['no_flag', 'flag']:
                f.write(f"\n{display_names[cls]}:\n")
                f.write(f"  Precision: {class_report[cls]['precision']:.4f}\n")
                f.write(f"  Recall: {class_report[cls]['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_report[cls]['f1-score']:.4f}\n")
            
            f.write("\nTiming Metrics:\n")
            f.write("--------------\n")
            f.write(f"Average batch time: {timing_stats['avg_batch_time']:.2f}ms\n")
            f.write(f"Batch time std: {timing_stats['std_batch_time']:.2f}ms\n")
            f.write(f"Average inference time per image: {timing_stats['avg_inference_time']:.2f}ms\n")
            f.write(f"Inference time std: {timing_stats['std_inference_time']:.2f}ms\n")
            f.write(f"Frames per second: {timing_stats['fps']:.2f}\n")
            f.write(f"Total images processed: {timing_stats['total_images']}\n")
            f.write(f"Total processing time: {timing_stats['total_time']:.2f}s\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write("----------------\n")
            f.write("True \\ Pred  |  No Flag  |  Flag\n")
            f.write(f"No Flag      |  {cm[0][0]:8d}  |  {cm[0][1]:5d}\n")
            f.write(f"Flag         |  {cm[1][0]:8d}  |  {cm[1][1]:5d}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python flag_metrics.py <model_path> <test_data_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    evaluator = FlagMetricsEvaluator(model_path, test_data_path)
    evaluator.run_evaluation()
    print("Evaluation complete. Results saved in metrics/ directory.")

if __name__ == "__main__":
    main()