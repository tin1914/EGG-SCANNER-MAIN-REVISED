from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import tempfile
import base64

app = Flask(__name__)

# Load the model once at startup
model = load_model('best_egg_quality_model.h5')

feature_names = [
    "Yolk Score",
    "White Score"
]

def preprocess_image(image_bytes):
    # Decode image from bytes
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Check if file is missing or corrupted.")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        img = preprocess_image(image_bytes)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0]
        prediction = np.clip(prediction, 0, 1)  # keep values between 0 and 1

        yolk_score = prediction[0] * 100
        white_score = prediction[1] * 100
        egg_quality_score = (yolk_score + white_score) / 2

        if egg_quality_score >= 80:
            quality = "GOOD"
        elif egg_quality_score >= 60:
            quality = "MEDIUM"
        else:
            quality = "BAD"

        result = {
            'yolk_score': round(yolk_score, 2),
            'white_score': round(white_score, 2),
            'egg_quality_score': round(egg_quality_score, 2),
            'quality': quality
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-tray', methods=['POST'])
def predict_tray():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read the file into memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Import tray prediction functions
        from tray_predict import detect_eggs, predict_egg_quality

        # Run prediction
        boxes = detect_eggs(image)
        egg_count = len(boxes)
        results = []
        for i, box in enumerate(boxes, start=1):
            result = predict_egg_quality(image, box)
            x, y, w, h = box
            yolk, white = result["scores"]
            label = result["label"]
            results.append({
                'egg_number': i,
                'yolk_score': round(yolk, 2),
                'white_score': round(white, 2),
                'overall_quality': round(result['egg_quality'], 2),
                'label': label
            })

        # Encode the annotated image as base64
        annotated = image.copy()
        for i, box in enumerate(boxes, start=1):
            result = predict_egg_quality(image, box)
            x, y, w, h = box
            color = result["color"]
            cv2.rectangle(annotated, (x,y), (x+w,y+h), color, 2)
            cv2.putText(annotated, f"Egg{i}: {result['label']} ({result['egg_quality']:.1f}%)",
                        (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'egg_count': egg_count,
            'results': results,
            'annotated_image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
