from flask import Blueprint, request, Response
import cv2
import uuid
import os

routes = Blueprint("routes", __name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@routes.route("/analyze-handwriting", methods=["POST"])
def analyze_handwriting():
    if "image" not in request.files:
        return {"error": "No image uploaded"}, 400
    
    # Save upload with UUID
    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.png"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    
    try:
        # Process using your exact pipeline logic
        annotated_img = process_handwriting(input_path)
        
        # Convert to PNG bytes
        _, img_bytes = cv2.imencode(".png", annotated_img)
        
        # Return as image/png
        return Response(img_bytes.tobytes(), mimetype="image/png")
        
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
