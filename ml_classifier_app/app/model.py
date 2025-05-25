import cv2
import numpy as np
from ultralytics import YOLO
import os

MODEL_PATH = "models/best.onnx"
model = YOLO(MODEL_PATH, task="detect")

def remove_lines(img):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    img = cv2.subtract(img, detect_horizontal)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    img = cv2.subtract(img, detect_vertical)
    return img

def preprocess_real_handwriting_image(img_path, output_size=(640, 640)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    binary = remove_lines(binary)

    h, w = binary.shape
    scale = min(output_size[0] / h, output_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(binary, (new_w, new_h))

    canvas = np.zeros(output_size, dtype=np.uint8)
    y_offset = (output_size[0] - new_h) // 2
    x_offset = (output_size[1] - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas

def highlight_digits(original_img_path, confidence_threshold=0.6):
    preprocessed = preprocess_real_handwriting_image(original_img_path)
    temp_pre_path = original_img_path + "_pre.png"
    cv2.imwrite(temp_pre_path, preprocessed)

    original_img = cv2.imread(original_img_path)
    highlighted_img = original_img.copy()
    orig_h, orig_w = original_img.shape[:2]

    results = model.predict(temp_pre_path, imgsz=640, show_conf=False, save = False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            if cls == 1 and conf < confidence_threshold:
                continue

            scale = min(640 / orig_w, 640 / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            x_offset = (640 - new_w) // 2
            y_offset = (640 - new_h) // 2

            x1 = int((x1 - x_offset) / scale)
            y1 = int((y1 - y_offset) / scale)
            x2 = int((x2 - x_offset) / scale)
            y2 = int((y2 - y_offset) / scale)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            overlay = highlighted_img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.3, highlighted_img, 0.7, 0, highlighted_img)
            cv2.rectangle(highlighted_img, (x1, y1), (x2, y2), color, 2)

    os.remove(temp_pre_path)
    return highlighted_img
