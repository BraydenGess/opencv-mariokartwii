import cv2
import numpy as np
from pathlib import Path
import os
# import glob

class TemplateFlagDetector:
    def __init__(self, template_path=None):
        # Load the checkered flag template
        if template_path is None:
            template_path = Path('templates/checkered_flag.png')
        
        self.template = cv2.imread(str(template_path))
        if self.template is None:
            raise ValueError(f"Could not load template from {template_path}")
            
        # Convert template to grayscale
        self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # Define ROI for flag detection with some padding for variations
        # Original coordinates: (274, 109, 72, 45)
        self.flag_roi = (250, 75, 150, 150)  # Expanded region to handle variations
        
        # Threshold for detection confidence
        self.confidence_threshold = 0.7
        
        # Scale range for template matching (handle size variations)
        # Since we know the expected size is around 72x45,
        # we can adjust the scale range accordingly
        self.scale_range = np.linspace(0.8, 1.2, 5)
        
    def create_binary_mask(self, image):
        """Create a binary mask highlighting black and white checkered pattern"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle lighting variations
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # C constant
        )
        
        return binary
        
    def detect_flag(self, frame):
        """
        Detect checkered flag in the frame using template matching.
        Returns (is_detected, confidence)
        """
        # Extract ROI with padding
        x, y, w, h = self.flag_roi
        roi = frame[y:y+h, x:x+w]
        
        # Create binary mask of ROI
        roi_binary = self.create_binary_mask(roi)
        
        best_confidence = 0
        best_scale = 1.0
        
        # Try different scales with finer granularity
        for scale in np.linspace(0.8, 1.2, 9):  # More scale steps
            # Resize template
            width = int(self.template.shape[1] * scale)
            height = int(self.template.shape[0] * scale)
            resized_template = cv2.resize(self.template, (width, height))
            
            # Try multiple methods and take best result
            methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
            for method in methods:
                result = cv2.matchTemplate(
                    roi_binary,
                    resized_template,
                    method
                )
                
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_scale = scale
        
        # Adjust confidence threshold based on testing
        self.confidence_threshold = 0.6  # Lowered threshold
        
        return best_confidence > self.confidence_threshold, best_confidence
        
    def visualize_detection(self, frame, is_detected, confidence):
        """
        Draw detection visualization on frame.
        Returns annotated frame.
        """
        vis_frame = frame.copy()
        
        # Draw ROI
        x, y, w, h = self.flag_roi
        color = (0, 255, 0) if is_detected else (0, 0, 255)
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
        
        # Add confidence text
        cv2.putText(
            vis_frame,
            f"Flag Confidence: {confidence:.2f}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        
        return vis_frame

def main():
    """Test the template detector with flag dataset"""
    detector = TemplateFlagDetector()
    
    # Test on flag dataset
    dataset_path = "Images/flag_data"
    
    # Process flag images
    flag_path = Path(dataset_path) / 'flag'
    no_flag_path = Path(dataset_path) / 'no_flag'
    
    total_images = 0
    correct_detections = 0
    
    # Process flag images (should detect flag)
    if flag_path.exists():
        for img_file in flag_path.glob('*.png'):
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue
                
            is_detected, confidence = detector.detect_flag(frame)
            
            # Draw and save visualization
            vis_frame = detector.visualize_detection(frame, is_detected, confidence)
            output_path = Path('metrics/visualizations') / f"flag_{img_file.stem}_detected.png"
            cv2.imwrite(str(output_path), vis_frame)
            
            total_images += 1
            if is_detected:
                correct_detections += 1
            
            print(f"Flag image {img_file.name}: {'✓' if is_detected else '✗'} ({confidence:.2f})")
    
    # Process no_flag images (should not detect flag)
    if no_flag_path.exists():
        for img_file in no_flag_path.glob('*.png'):
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue
                
            is_detected, confidence = detector.detect_flag(frame)
            
            # Draw and save visualization
            vis_frame = detector.visualize_detection(frame, is_detected, confidence)
            output_path = Path('metrics/visualizations') / f"no_flag_{img_file.stem}_detected.png"
            cv2.imwrite(str(output_path), vis_frame)
            
            total_images += 1
            if not is_detected:
                correct_detections += 1
                
            print(f"No flag image {img_file.name}: {'✓' if not is_detected else '✗'} ({confidence:.2f})")
    
    # Print summary
    accuracy = correct_detections / total_images if total_images > 0 else 0
    print(f"\nResults:")
    print(f"Total images: {total_images}")
    print(f"Correct detections: {correct_detections}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    Path('metrics/visualizations').mkdir(parents=True, exist_ok=True)
    main()