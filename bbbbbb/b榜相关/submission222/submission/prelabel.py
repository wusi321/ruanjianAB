# -*- coding: utf-8 -*-
"""
Competition predict.py entry point.
Loads Paddle Inference model from model/, detects fire regions,
outputs only the largest firebig box per image in competition JSON format.
Also supports pre-labeling mode (LabelMe JSON) when called as prelabel.py.
"""

import os
import sys
import cv2
import json
import glob
import numpy as np


# =========================================================
# Paddle Inference model (lazy-load, shared across all calls)
# =========================================================

_predictor = None
_input_names = None
_output_names = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def _load_model():
    """Lazy-load the Paddle Inference model. Returns True on success."""
    global _predictor, _input_names, _output_names
    if _predictor is not None:
        return True

    model_file = os.path.join(MODEL_DIR, "model.pdmodel")
    params_file = os.path.join(MODEL_DIR, "model.pdiparams")
    if not os.path.exists(model_file) or not os.path.exists(params_file):
        print("[Model] Model files not found in {}. Falling back to HSV.".format(MODEL_DIR))
        return False

    try:
        from paddle.inference import Config, create_predictor

        config = Config(model_file, params_file)
        config.enable_use_gpu(2048, 0)
        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.disable_glog_info()

        _predictor = create_predictor(config)
        _input_names = _predictor.get_input_names()
        _output_names = _predictor.get_output_names()
        print("[Model] Loaded from {} (inputs: {}, outputs: {})".format(
            MODEL_DIR, _input_names, _output_names))
        return True
    except Exception as e:
        print("[Model] Failed to load: {}".format(e))
        return False


def _preprocess_image(img):
    """
    Preprocess image to model input format.
    Matches my_ppyoloe.yml: resize 640x640, keep_ratio=false,
    normalize with ImageNet stats, HWC -> CHW.
    NOTE: PaddleDetection Decode produces BGR images and PP-YOLOE+
    pretrained weights are trained on BGR. Keep BGR, DO NOT convert to RGB.
    """
    h, w = img.shape[:2]

    # Resize to 640x640, no keep_ratio, bilinear (interp=2)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Keep BGR (PaddleDetection convention), scale to [0, 1], normalize
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW, add batch dim
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Scale factor for coordinate rescaling inside model NMS
    scale = np.array([[640.0 / h, 640.0 / w]], dtype=np.float32)

    return img, scale


def _model_infer(img):
    """
    Run Paddle Inference model on a single image (BGR np.ndarray).
    Returns list of [x, y, w, h, score] detections, or None if model unavailable.
    Coordinates are in original image pixel space.
    """
    if not _load_model():
        return None

    img_tensor, scale = _preprocess_image(img)

    # Feed inputs dynamically by name
    for name in _input_names:
        handle = _predictor.get_input_handle(name)
        if name in ("image", "inputs", "x"):
            handle.reshape(img_tensor.shape)
            handle.copy_from_cpu(img_tensor)
        elif name == "scale_factor":
            handle.reshape(scale.shape)
            handle.copy_from_cpu(scale)

    _predictor.run()

    # Parse output: [N, 6] or [1, N, 6], format [class_id, score, x1, y1, x2, y2]
    # Some exports also use 7-column: [batch_id, class_id, score, x1, y1, x2, y2]
    output = _predictor.get_output_handle(_output_names[0]).copy_to_cpu()
    if output.ndim == 3:
        output = output[0]
    if output.size == 0:
        return []

    dets = []
    for row in output:
        row = row.tolist()
        if len(row) == 6:
            cls_id, score, x1, y1, x2, y2 = row
        elif len(row) == 7:
            batch_id, cls_id, score, x1, y1, x2, y2 = row
            if batch_id < 0:
                continue
        else:
            continue
        if score < 0.15:
            continue
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        dets.append([float(x1), float(y1), float(bw), float(bh), float(score)])
    return dets


# =========================================================
# HSV fallback (used when model is not available)
# =========================================================

def _predict_fire_region_hsv(img):
    """HSV color-based fire detection. Returns single largest fire shape, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 50, 150])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

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

    return {
        "label": "firebig",
        "points": [[x, y], [x + w, y + h]],
    }


# =========================================================
# Main detection entry (model first, HSV fallback)
# =========================================================

def predict_fire_region(img_path, img=None):
    """
    Run model inference to detect the largest fire region.
    Falls back to HSV if model is not available.
    Returns {"label": "firebig", "points": [[x1,y1],[x2,y2]]} or None.
    """
    if img is None:
        img = cv2.imread(img_path)
    if img is None:
        return None

    # Try model inference
    dets = _model_infer(img)

    if dets is None:
        # Model not available, fall back to HSV
        return _predict_fire_region_hsv(img)

    if not dets:
        return None

    # Take the largest detection by bbox area (firebig = largest fire)
    largest = max(dets, key=lambda d: d[2] * d[3])
    x, y, w, h = largest[0], largest[1], largest[2], largest[3]

    if w < 5 or h < 5:
        return None

    return {
        "label": "firebig",
        "points": [[x, y], [x + w, y + h]],
    }


# =========================================================
# Output helpers
# =========================================================

def save_as_labelme_json(img_path, shape, output_dir):
    """Save as LabelMe JSON format (can be opened directly in LabelMe)."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2] if img is not None else (480, 640)

    data = {
        "version": "5.0.1",
        "flags": {},
        "imagePath": os.path.basename(img_path),
        "imageHeight": h,
        "imageWidth": w,
        "imageDepth": 3,
        "shapes": [shape] if shape else [],
    }

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    out_path = os.path.join(output_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    status = "has pre-label" if shape else "no detection"
    print("[{}] {} -> {}".format(status, img_path, out_path))


def batch_prelabel(image_dir, output_dir):
    """Batch pre-labeling mode."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    images = sorted(images)

    print("Found {} images, starting pre-labeling...\n".format(len(images)))
    for img_path in images:
        shape = predict_fire_region(img_path)
        save_as_labelme_json(img_path, shape, output_dir)

    print("\nDone! Output dir: {}".format(output_dir))
    print("Open the image folder with LabelMe, JSON files will auto-load as pre-labels.")


def competition_predict(data_txt, result_json):
    """
    Competition inference entry point.
    Reads image path list from data_txt, runs model inference on each image,
    outputs at most 1 largest firebig box per image in competition JSON format.
    """
    with open(data_txt, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    results = []
    for img_path in image_paths:
        shape = predict_fire_region(img_path)
        if shape is None:
            continue

        points = shape["points"]
        x, y = points[0]
        x2, y2 = points[1]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        results.append({
            "image_id": img_name,
            "type": 1,
            "x": float(x),
            "y": float(y),
            "width": float(x2 - x),
            "height": float(y2 - y),
            "segmentation": []
        })

    output = {"result": results}
    os.makedirs(os.path.dirname(result_json) or ".", exist_ok=True)
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nDone. {} images -> {} detections.".format(len(image_paths), len(results)))
    print("Saved to: {}".format(result_json))


# =========================================================
# Entry point
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
        print("    Example: python predict.py test_list.txt result.json")
        print("  Pre-label:   python prelabel.py <image_dir> <output_json_dir>")
        print("    Example: python prelabel.py D:\\images D:\\prelabels")
