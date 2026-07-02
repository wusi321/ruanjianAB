# -*- coding: utf-8 -*-
"""
Competition predict.py entry point.
Loads Paddle Inference model from model/, detects fire regions,
outputs only 1 firebig box per image in competition JSON format.
"""

import os
import sys
import time
import cv2
import json
import glob
import numpy as np


# =========================================================
# Configuration
# =========================================================

SCORE_THRESH = 0.25          # Lower = more detections (try 0.05~0.15)
INPUT_SIZE = 640
CLASS_ID = 1

# =========================================================
# Paddle Inference model (lazy-load)
# =========================================================

_predictor = None
_input_names = None
_output_names = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def _load_model():
    global _predictor, _input_names, _output_names
    if _predictor is not None:
        return True
    model_file = os.path.join(MODEL_DIR, "model.pdmodel")
    params_file = os.path.join(MODEL_DIR, "model.pdiparams")
    if not os.path.exists(model_file) or not os.path.exists(params_file):
        print("[Model] Not found in {}".format(MODEL_DIR))
        return False
    try:
        from paddle.inference import Config, create_predictor
        config = Config(model_file, params_file)
        config.enable_use_gpu(2048, 0)
        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(True)
        config.disable_glog_info()
        _predictor = create_predictor(config)
        _input_names = _predictor.get_input_names()
        _output_names = _predictor.get_output_names()
        print("[Model] Loaded (inputs={}, outputs={})".format(_input_names, _output_names))
        return True
    except Exception as e:
        print("[Model] Error: {}".format(e))
        return False


def _preprocess(img):
    """Resize 640x640, normalize BGR with ImageNet stats, HWC->CHW."""
    raw_h, raw_w = img.shape[:2]
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    scale_y = float(INPUT_SIZE) / float(raw_h)
    scale_x = float(INPUT_SIZE) / float(raw_w)
    return img, scale_x, scale_y, raw_w, raw_h


def _predict_image(img):
    """
    Run model on one BGR image.
    Returns list of dicts: [{"x","y","width","height","score"}, ...]
    or empty list if no detections.
    """
    if not _load_model():
        return []

    input_data, scale_x, scale_y, raw_w, raw_h = _preprocess(img)
    scale_factor = np.array([[scale_y, scale_x]], dtype=np.float32)

    for name in _input_names:
        tensor = _predictor.get_input_handle(name)
        if name in ("image", "inputs", "x"):
            tensor.copy_from_cpu(input_data)
        elif name == "scale_factor":
            tensor.copy_from_cpu(scale_factor)

    _predictor.run()

    # Collect output boxes
    boxes = None
    for name in _output_names:
        out = _predictor.get_output_handle(name).copy_to_cpu()
        if out.ndim == 3:
            out = out[0]
        if out.ndim == 2 and out.shape[1] in (6, 7):
            boxes = out
            break
    if boxes is None or boxes.size == 0:
        return []

    results = []
    for row in boxes:
        row = row.tolist()
        if len(row) == 6:
            cls_id, score, x1, y1, x2, y2 = row
        elif len(row) == 7:
            batch_id, cls_id, score, x1, y1, x2, y2 = row
            if batch_id < 0:
                continue
        else:
            continue

        if score < SCORE_THRESH:
            continue

        # Clip to image boundaries
        x1 = max(0.0, float(x1))
        y1 = max(0.0, float(y1))
        x2 = min(float(raw_w) - 1, float(x2))
        y2 = min(float(raw_h) - 1, float(y2))
        w = x2 - x1
        h = y2 - y1

        # ---- oversized box filter (from predict3333.py) ----
        if w <= 2 or h <= 2:
            continue
        if w > raw_w * 0.95:
            continue
        if h > raw_h * 0.95:
            continue
        if w * h > raw_w * raw_h * 0.95:
            continue

        results.append({
            "x": round(x1, 2),
            "y": round(y1, 2),
            "width": round(w, 2),
            "height": round(h, 2),
            "score": float(score),
        })

    if not results:
        return []

    # Keep only the largest box by area (firebig = largest fire)
    best = max(results, key=lambda r: r["width"] * r["height"])
    best.pop("score", None)
    return [best]


# =========================================================
# HSV fallback
# =========================================================

def _predict_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 150]), np.array([35, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    if w < 20 or h < 20:
        return None
    return {"label": "firebig", "points": [[x, y], [x + w, y + h]]}


# =========================================================
# Main prediction entry
# =========================================================

def predict_fire_region(img_path, img=None):
    if img is None:
        img = cv2.imread(img_path)
    if img is None:
        return None

    result = _predict_image(img)

    if result is None:
        return _predict_hsv(img)

    if not result:
        return None

    r = result[0]
    return {
        "label": "firebig",
        "points": [[r["x"], r["y"]], [r["x"] + r["width"], r["y"] + r["height"]]],
    }


# =========================================================
# LabelMe pre-labeling helpers
# =========================================================

def save_as_labelme_json(img_path, shape, output_dir):
    img = cv2.imread(img_path)
    h, w = img.shape[:2] if img is not None else (480, 640)
    data = {
        "version": "5.0.1", "flags": {},
        "imagePath": os.path.basename(img_path),
        "imageHeight": h, "imageWidth": w, "imageDepth": 3,
        "shapes": [shape] if shape else [],
    }
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    out_path = os.path.join(output_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("[{}] {} -> {}".format("OK" if shape else "NONE", img_path, out_path))


def batch_prelabel(image_dir, output_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    images = sorted(images)
    print("Found {} images\n".format(len(images)))
    for img_path in images:
        shape = predict_fire_region(img_path)
        save_as_labelme_json(img_path, shape, output_dir)
    print("\nDone: {}".format(output_dir))


# =========================================================
# Competition entry point
# =========================================================

def competition_predict(data_txt, result_json):
    with open(data_txt, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    print("Total images: {}".format(len(image_paths)))
    t_start = time.time()

    all_results = []
    for idx, img_path in enumerate(image_paths):
        try:
            shape = predict_fire_region(img_path)
            if shape is not None:
                points = shape["points"]
                x, y = points[0]
                x2, y2 = points[1]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                all_results.append({
                    "image_id": img_name,
                    "type": CLASS_ID,
                    "x": float(x),
                    "y": float(y),
                    "width": float(x2 - x),
                    "height": float(y2 - y),
                    "segmentation": []
                })
        except Exception as e:
            print("ERROR [{}]: {}".format(img_path, e))

        if (idx + 1) % 500 == 0:
            print("Progress: {}/{}".format(idx + 1, len(image_paths)))

    elapsed = time.time() - t_start
    fps = len(image_paths) / elapsed if elapsed > 0 else 0
    print("\n========================================")
    print("FPS: {:.2f}  |  Time: {:.2f}s  |  Dets: {}".format(fps, elapsed, len(all_results)))
    print("========================================")

    output = {"result": all_results}
    os.makedirs(os.path.dirname(result_json) or ".", exist_ok=True)
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Saved: {}".format(result_json))


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[1].lower().endswith(".txt"):
            competition_predict(sys.argv[1], sys.argv[2])
        else:
            batch_prelabel(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  Competition: python predict.py <data_txt> <result_json>")
        print("  Pre-label:   python prelabel.py <image_dir> <output_json_dir>")
