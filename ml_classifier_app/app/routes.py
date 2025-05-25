from flask import Blueprint, request, Response
import uuid
import os
import cv2

from models import highlight_digits

routes = Blueprint("routes", __name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@routes.route("/analyze-handwriting", methods=["POST"])
def analyze_handwriting():
    if "image" not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.png"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    try:
        annotated_img = highlight_digits(input_path)
        _, img_bytes = cv2.imencode(".png", annotated_img)
        return Response(img_bytes.tobytes(), mimetype="image/png")
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
