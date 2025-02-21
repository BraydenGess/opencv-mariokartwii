"""
Course Detection Module for Mario Kart Screenshots
===============================================

This module implements a comprehensive course detection system for Mario Kart gameplay
screenshots. It combines OCR (Optical Character Recognition) with computer vision and
deep learning techniques to identify race courses from in-game images.

Key Components:
-------------
- OCR-based text detection: Uses Tesseract with multiple PSM modes for robust text recognition
- Flag detection: Neural network model to validate race screenshots via checkpoint flags
- Image preprocessing: Custom filters and transformations to optimize text extraction
- Multi-stage validation: Combines multiple detection methods for higher accuracy

The system handles various challenges including:
- Variable text positioning and formatting
- Different lighting conditions and motion blur
- False positives from UI elements and decorative text
- Multiple languages and character sets

Usage:
-----
python course_detector.py <flag_model_path>

The flag_model_path argument should point to a trained PyTorch model for flag detection.
The system will process images and output course identifications along with confidence scores.

This is part of a larger Mario Kart course recognition system that enables automated
gameplay analysis and statistics tracking.

Author: Max White
Date: February 2025
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import time
import sys
import matplotlib.pyplot as plt

def display(image):
    """
    Display an image using matplotlib. Accepts either an image path (str) or numpy array.
    If interactive display isn't available, saves to a debug file instead.
    """
    try:
        # Try to switch to interactive backend if not already
        if plt.get_backend() == 'agg':
            plt.switch_backend('TkAgg')
        
        dpi = 80
        
        # Handle both file paths and numpy arrays
        if isinstance(image, str):
            im_data = plt.imread(image)
        else:
            im_data = image

        height, width = im_data.shape[:2]
        
        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        # Display the image.
        ax.imshow(im_data, cmap='gray')

        plt.show()
    except:
        # If interactive display fails, save to file instead
        if isinstance(image, str):
            print(f"Image already saved at: {image}")
        else:
            cv2.imwrite('debug_display.png', image)
            #print("Saved debug image to: debug_display.png")

class FlagDetector(nn.Module):
    def __init__(self):
        super(FlagDetector, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)
    
class SimpleFlagDetector(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleFlagDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64 * 8 * 8, 64),  # For 64x64 input
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CourseDetector:
    def __init__(self, flag_model_path, use_template=False, psm=6): # PSM 6 and 9 are the best eh idk actually
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
        
        #self.tesseract_config = '--psm 7 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'" -l eng --dpi 300'
        
        # Flag detection setup
        if use_template:
            # Use template matching approach
            from template_flag_detector import TemplateFlagDetector
            self.flag_detector = TemplateFlagDetector(flag_model_path)
            self.flag_roi = self.flag_detector.flag_roi
        else:
            # Use CNN approach
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.flag_detector = SimpleFlagDetector().to(self.device)
            self.flag_detector.load_state_dict(torch.load(flag_model_path, map_location=self.device))
            self.flag_detector.eval()  # Make sure we're in eval mode
            
            # Transform for flag detection
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.flag_roi = (274, 109, 72, 45)  # Match training ROI exactly
        
        self.confidence_threshold = 0.6
        self.psm = psm

    def _detect_flag(self, frame):
        """Detect flag in the ROI region - matched exactly with training"""
        try:
            # Extract ROI using exact same coordinates as training
            x, y, w, h = self.flag_roi
            roi = frame[y:y+h, x:x+w]
            
            # Debug: Save ROI for inspection
            cv2.imwrite('debug_roi.png', roi)
            
            # Convert BGR to RGB (same as training)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image (same as training Dataset class)
            roi_pil = Image.fromarray(roi_rgb)
            
            # Debug: Save PIL image for inspection
            roi_pil.save('debug_pil.png')
            
            # Apply same transforms as training
            input_tensor = self.transform(roi_pil)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run detection
            with torch.no_grad():
                output = self.flag_detector(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence = probabilities[0][1].item()
                prediction = confidence > self.confidence_threshold
                
                # Debug: Print prediction details
                # print(f"Raw output: {output}")
                # print(f"Probabilities: {probabilities}")
                # print(f"Confidence: {confidence}")
                # print(f"Prediction: {prediction}")
            
            return prediction, confidence
            
        except Exception as e:
            # print(f"Error in flag detection: {e}")
            return False, 0.0
        
    def _noise_removal(self, image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)
    
    def _thin_font(self, image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)

    
    def _extract_course_name(self, frame):
        """Extract all text from the frame using OCR."""
        if frame is None or frame.size == 0:
            return ""
        
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Threshold the image
            _, binary = cv2.threshold(gray, 150, 230, cv2.THRESH_BINARY)

            # Remove noise
            img = self._noise_removal(binary)

            # Erode 
            img = self._thin_font(img)

            x, y, w, h = self.text_roi

            # display(img[y:y+h, x:x+w])

            # Get course text from the binary image
            custom_config = f'--psm {self.psm}'  # Use the instance PSM value
            course_text = pytesseract.image_to_string(img[y:y+h, x:x+w], config=custom_config).strip()
            
            # Debug: Print the extracted text
            # print(f"OCR extracted text: {course_text}")
            
            return course_text
            
        except Exception as e:
            # print(f"Error in text extraction: {e}")
            return ""
    
    def _match_course_name(self, detected_text):
        """Match detected text to known course names."""
        if not detected_text:
            return None, 0.0
            
        best_match = None
        best_ratio = 0
        
        from difflib import SequenceMatcher
        for course in self.course_names:
            ratio = SequenceMatcher(None, detected_text.lower(), course.lower()).ratio()
            if ratio > best_ratio and ratio > 0.5:
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
        
        # Extract ROI for course name text
        # x, y, w, h = self.text_roi
        # course_roi = frame[y:y+h, x:x+w]
        
        # Get all text detections and potential course name
        course_text = self._extract_course_name(frame)
        
        # Store raw text for logging (before any processing)
        self.last_raw_text = course_text
        
        if not flag_detected:
            return None, 0, None
            
        # If flag found, try to match course name
        course_name, text_confidence = self._match_course_name(course_text)
        
        # Debug print
        # print(f"Raw text detected: {course_text}")
        # print(f"Matched course: {course_name}")
        
        # Return course name, combined confidence, and raw text
        if course_name:
            confidence = (flag_confidence + text_confidence) / 2
            return course_name, confidence, course_text
        return None, 0, course_text

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