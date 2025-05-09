import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from ultralytics import YOLO
import os

# Initialize models
model = YOLO("models/best3.onnx", task="classify")

def process_handwriting(image_path):
    # 1. Load Image (with validation)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # 2. Preprocessing Pipeline 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 10)
    cleaned = remove_horizontal_lines(binary)

    # 3. Character Extraction with Tesseract
    data = pytesseract.image_to_data(cleaned, config='--psm 6', output_type=Output.DICT)
    
    # 4. Process each character 
    vis_img = img.copy()
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if not text:  # Skip empty strings
            continue
            
        # Get coordinates
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        avg_char_width = max(1, w // len(text))
        
        # Process each character in the word
        for j, char in enumerate(text):
            if not char.isalnum():  # Skip non-alphanumeric (matches  'continue' logic)
                continue
                
            char_x = x + j * avg_char_width
            
            # Crop single character (with boundary checks)
            char_img = img[
                max(0, y):min(img.shape[0], y+h),
                max(0, char_x):min(img.shape[1], char_x+avg_char_width)
            ]
            
            # Classify (only if we got a valid crop)
            if char_img.size > 0:
                result = model.predict(char_img, imgsz=224, verbose=False)[0]
                if hasattr(result, 'probs') and result.probs.top1 == 0:  # Assuming class 0 is dyslexic
                    # Draw rectangle (matches highlighting approach)
                    cv2.rectangle(vis_img, 
                                (char_x, y),
                                (char_x + avg_char_width, y + h),
                                (0, 0, 255),  # Red for dyslexic
                                2)
    
    return vis_img

# Your exact preprocessing functions
def remove_horizontal_lines(binary_img):
    width = binary_img.shape[1]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 30), 1))
    horiz_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    return cv2.subtract(binary_img, horiz_lines)
