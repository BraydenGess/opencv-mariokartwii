# python3 course_detector.py <flag_model_path>

import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import time
import sys

class FlagDetector(nn.Module):
    def __init__(self):
        super(FlagDetector, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

class CourseDetector:
    def __init__(self, flag_model_path, use_template=False):
        self.text_roi = (1000, 900, 800, 100)  # Keep text ROI for course names
        
        # Known course names for matching (needs update for more courses)
        self.course_names = [
            "Luigi Circuit",
            "Moo Moo Meadows", 
            "Mushroom Gorge",
            "Toad's Factory",
            "Mario Circuit",
            "Coconut Mall",
            "DK Summit",
            "Wario's Gold Mine",
            "Daisy Circuit",
            "Koopa Cape",
            "Maple Treeway",
            "Grumble Volcano",
            "Dry Dry Ruins",
            "Moonview Highway",
            "Bowser's Castle",
            "Rainbow Road",
            "GCN Peach Beach",
            "DS Yoshi Falls",
            "SNES Ghost Valley 2",
            "N64 Mario Raceway",
            "N64 Sherbet Land",
            "GBA Shy Guy Beach",
            "DS Delfino Square",
            "GCN Waluigi Stadium",
            "DS Desert Hills",
            "GBA Bowser Castle 3",
            "N64 DK's Jungle Parkway",
            "GCN Mario Circuit",
            "SNES Mario Circuit 3",
            "DS Peach Gardens",
            "GCN DK Mountain",
            "N64 Bowser's Castle"
        ]
        
        self.tesseract_config = '--psm 7 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'" -l eng --dpi 300'
        
        # Flag detection setup
        if use_template:
            # Use template matching approach
            from template_flag_detector import TemplateFlagDetector
            self.flag_detector = TemplateFlagDetector(flag_model_path)
            self.flag_roi = self.flag_detector.flag_roi
        else:
            # Use CNN approach
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.flag_detector = FlagDetector().to(self.device)
            self.flag_detector.load_state_dict(torch.load(flag_model_path, map_location = self.device))
            self.flag_detector.eval()
            
            # Transform for flag detection
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            self.flag_roi = (250, 75, 150, 150)  # Keep consistent with template detector
        
        self.confidence_threshold = 0.7
        
    def _detect_flag(self, frame):
        """Detect flag in the ROI region"""
        if hasattr(self.flag_detector, 'detect_flag'):
            # Use template detector's method
            return self.flag_detector.detect_flag(frame)
        else:
            # Use CNN detection
            x, y, w, h = self.flag_roi
            roi = frame[y:y+h, x:x+w]
            
            # Convert ROI to PIL Image
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            
            # Prepare for model
            input_tensor = self.transform(roi_pil)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run detection
            with torch.no_grad():
                output = self.flag_detector(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence = probabilities[0][1].item()
            
            return confidence > self.confidence_threshold, confidence
    
    def _extract_course_name(self, frame):
        """Extract all text from the frame using OCR."""
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Get detailed OCR data including bounding boxes
        ocr_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
        
        # Collect all detected text and their positions
        text_detections = []
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            # Filter empty results and those with low confidence
            if int(ocr_data['conf'][i]) > 0:  # You can adjust this confidence threshold
                text = ocr_data['text'][i].strip()
                if text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    conf = int(ocr_data['conf'][i])
                    text_detections.append({
                        'text': text,
                        'box': (x, y, w, h),
                        'confidence': conf
                    })
        
        # Also get the course name from the specific ROI for matching
        x, y, w, h = self.text_roi
        roi_region = binary[y:y+h, x:x+w]
        course_text = pytesseract.image_to_string(roi_region, config=self.tesseract_config).strip()
        
        return text_detections, course_text
    
    def _match_course_name(self, detected_text):
        """Match detected text to known course names."""
        if not detected_text:
            return None, 0.0
            
        best_match = None
        best_ratio = 0
        
        from difflib import SequenceMatcher
        for course in self.course_names:
            ratio = SequenceMatcher(None, detected_text.lower(), course.lower()).ratio()
            if ratio > best_ratio and ratio > 0.7:
                best_ratio = ratio
                best_match = course
                
        return best_match, best_ratio
    
    def detect_course(self, frame):
        """
        Main detection method.
        Returns (course_name, confidence, text_detections)
        """
        # Check for flag in entire frame
        flag_detected, flag_confidence = self._detect_flag(frame)
        
        # Get all text detections and potential course name
        text_detections, course_text = self._extract_course_name(frame)
        
        if not flag_detected:
            return None, 0, text_detections
            
        # If flag found, try to match course name
        course_name, text_confidence = self._match_course_name(course_text)
        
        # For debugging
        if course_name:
            print(f"Flag confidence: {flag_confidence:.2f}")
            print(f"Detected course text: {course_text}")
            print(f"Matched course: {course_name}")
            
        # Return course name, combined confidence, and all text detections
        if course_name:
            confidence = (flag_confidence + text_confidence) / 2
            return course_name, confidence, text_detections
        return None, 0, text_detections

def main():
    """Test the detector with a video capture."""
    flag_model_path = sys.argv[1]
    cap = cv2.VideoCapture(0)  # or path to video file
    detector = CourseDetector(flag_model_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw ROIs for debugging
        x, y, w, h = detector.text_roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Time the detection
        start_time = time.time()
        course_name, confidence, text_detections = detector.detect_course(frame)
        process_time = time.time() - start_time
        
        # Display results
        if course_name:
            cv2.putText(frame, f"{course_name} ({confidence:.2f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {1/process_time:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()