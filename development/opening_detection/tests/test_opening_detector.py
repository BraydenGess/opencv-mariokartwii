import torch
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys

# Add parent directory to path to import OpeningDetector
sys.path.append(str(Path(__file__).parent.parent))
from opening_detector import OpeningDetector

class OpeningDetectorTester:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OpeningDetector().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def test_dataset(self, test_path, debug=False):
        """Test all images in the dataset directory structure"""
        test_path = Path(test_path)
        
        # Initialize results storage
        all_predictions = []
        true_labels = []
        prediction_details = []
        
        # Process opening images
        opening_path = test_path / 'opening'
        no_opening_path = test_path / 'no_opening'
        
        # Test opening images
        for img_path in opening_path.glob('*.png'):
            if debug:
                print(f"Processing opening image: {img_path}")
            
            frame = cv2.imread(str(img_path))
            is_opening, confidence = self.model.predict(frame)
            
            all_predictions.append(1 if is_opening else 0)
            true_labels.append(1)
            
            prediction_details.append({
                'image_path': str(img_path),
                'predicted': 'opening' if is_opening else 'no_opening',
                'actual': 'opening',
                'confidence': float(confidence)
            })

        # Test no_opening images
        for img_path in no_opening_path.glob('*.png'):
            if debug:
                print(f"Processing no_opening image: {img_path}")
            
            frame = cv2.imread(str(img_path))
            is_opening, confidence = self.model.predict(frame)
            
            all_predictions.append(1 if is_opening else 0)
            true_labels.append(0)
            
            prediction_details.append({
                'image_path': str(img_path),
                'predicted': 'opening' if is_opening else 'no_opening',
                'actual': 'no_opening',
                'confidence': float(confidence)
            })

        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(true_labels, all_predictions)),
            'precision': float(precision_score(true_labels, all_predictions)),
            'recall': float(recall_score(true_labels, all_predictions)),
            'f1': float(f1_score(true_labels, all_predictions))
        }
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, all_predictions)
        
        # Create report
        report = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'predictions': prediction_details
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path('test_results')
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        report_path = results_dir / f'opening_detector_results_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Opening', 'Opening'],
                    yticklabels=['No Opening', 'Opening'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = results_dir / f'confusion_matrix_{timestamp}.png'
        plt.savefig(cm_path)
        plt.close()
        
        # Print summary
        print("\nTest Results Summary:")
        print("-" * 50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"\nResults saved to: {report_path}")
        print(f"Confusion matrix plot saved to: {cm_path}")
        
        return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Opening Detector on a dataset')
    parser.add_argument('model_path', help='Path to the trained model')
    parser.add_argument('test_path', help='Path to test dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    tester = OpeningDetectorTester(args.model_path)
    tester.test_dataset(args.test_path, args.debug)

if __name__ == '__main__':
    main()
