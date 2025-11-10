
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "best_egg_quality_model.h5"
model = load_model(MODEL_PATH)

SAVE_DEBUG = True
EXPECTED_EGGS = 6

DP = 1.2
MIN_DIST_FRAC = 0.22
CANNY_HIGH = 120
ACCUM_THRESH = 22
R_MIN_FRAC = 0.085
R_MAX_FRAC = 0.22
PADDING_FRAC = 0.10


def preprocess_crop(crop):
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.float32) / 255.0
    return np.expand_dims(crop, axis=0)


def predict_egg_quality(image_bgr, box):
    x, y, w, h = box
    crop = image_bgr[y:y+h, x:x+w]
    preds = model.predict(preprocess_crop(crop), verbose=0)[0]
    preds = np.clip(preds, 0, 1)

    yolk = preds[0] * 100
    white = preds[1] * 100

    egg_quality = (yolk + white) / 2

    if egg_quality >= 55:
        label = "GOOD"
        color = (0, 255, 0)
    elif egg_quality >= 30:
        label = "MEDIUM"
        color = (0, 255, 255)
    else:
        label = "BAD"
        color = (0, 0, 255)

    return {
        "scores": (yolk, white),
        "egg_quality": egg_quality,
        "label": label,
        "color": color
    }


def boxes_from_circles(circles, shape):
    H, W = shape[:2]
    boxes = []
    for (cx, cy, r) in circles:
        pad = int(PADDING_FRAC * r)
        x0 = max(0, int(cx - r - pad))
        y0 = max(0, int(cy - r - pad))
        x1 = min(W, int(cx + r + pad))
        y1 = min(H, int(cy + r + pad))
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def hough_detect(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=60, sigmaSpace=60)
    smooth = cv2.medianBlur(smooth, 7)

    H, W = gray.shape[:2]
    min_dist = int(min(H, W) * MIN_DIST_FRAC)
    rmin = int(min(H, W) * R_MIN_FRAC)
    rmax = int(min(H, W) * R_MAX_FRAC)

    circles = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT, dp=DP, minDist=min_dist,
        param1=CANNY_HIGH, param2=ACCUM_THRESH,
        minRadius=rmin, maxRadius=rmax
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)

    kept = []
    for c in circles:
        if all(np.hypot(c[0]-k[0], c[1]-k[1]) >= min_dist * 0.9 for k in kept):
            kept.append(c)
        if len(kept) >= 12:
            break

    if SAVE_DEBUG:
        dbg = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
        for (cx, cy, r) in kept:
            cv2.circle(dbg, (cx,cy), r, (0,255,0), 2)
        cv2.imwrite("tray_debug_mask.jpg", dbg)

    kept = kept[:EXPECTED_EGGS]
    return boxes_from_circles(kept, image_bgr.shape)


def fallback_grid(image_bgr):
    H, W = image_bgr.shape[:2]
    m = 0.06
    x0, x1 = int(W*m), int(W*(1-m))
    y0, y1 = int(H*m), int(H*(1-m))
    w = (x1 - x0) // 3
    h = (y1 - y0) // 2
    boxes = []
    for r in range(2):
        for c in range(3):
            x = x0 + c*w
            y = y0 + r*h
            pad = int(0.12 * max(w, h))
            X0 = max(0, x + pad//2)
            Y0 = max(0, y + pad//2)
            X1 = min(W, x + w - pad//2)
            Y1 = min(H, y + h - pad//2)
            boxes.append((X0, Y0, X1-X0, Y1-Y0))
    return boxes


def detect_eggs(image_bgr):
    boxes = hough_detect(image_bgr)
    if len(boxes) < EXPECTED_EGGS:
        boxes += fallback_grid(image_bgr)
    return boxes[:EXPECTED_EGGS]


def run_tray_predict(image_path="tray.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print("Could not find image:", image_path)
        return

    boxes = detect_eggs(image)
    print(f"\nDetected {len(boxes)} eggs\n")

    annotated = image.copy()
    for i, box in enumerate(boxes, start=1):
        result = predict_egg_quality(image, box)
        x, y, w, h = box
        yolk, white = result["scores"]
        label = result["label"]
        color = result["color"]

        cv2.rectangle(annotated, (x,y), (x+w,y+h), color, 2)
        cv2.putText(annotated, f"Egg{i}: {label} ({result['egg_quality']:.1f}%)",
                    (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        print(f"Egg {i}: Yolk={yolk:.2f}%, White={white:.2f}%, Overall={result['egg_quality']:.2f}% → {label}")

    cv2.imwrite("tray_result.jpg", annotated)


# Run only when executed directly
if __name__ == "__main__":
    run_tray_predict()
