import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "best_egg_quality_model.h5"
model = load_model(MODEL_PATH)

feature_names = ["Yolk Score", "White Score"]


def preprocess_image(image_path):
    with open(image_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img


def run_single_predict(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]
    prediction = np.clip(prediction, 0, 1)

    yolk_score = prediction[0] * 100
    white_score = prediction[1] * 100
    egg_quality_score = (yolk_score + white_score) / 2

    print(f"\nResults for: {image_path}\n")
    print(f"- Yolk Score:  {yolk_score:.2f}%")
    print(f"- White Score: {white_score:.2f}%")
    print(f"----------------------------------")
    print(f"Overall Egg Quality Score: {egg_quality_score:.2f}%")

    if egg_quality_score >= 80:
        print("→ Verdict: GOOD quality egg")
    elif egg_quality_score >= 60:
        print("→ Verdict: MEDIUM quality egg")
    else:
        print("→ Verdict: BAD quality egg")

    print("\nPrediction complete.\n")

    return {
        "yolk": yolk_score,
        "white": white_score,
        "overall": egg_quality_score
    }


# Allow direct execution
if __name__ == "__main__":
    run_single_predict("fresh.jpg")  # default test image
