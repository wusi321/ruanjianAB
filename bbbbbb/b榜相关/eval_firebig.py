# -*- coding: utf-8 -*-
"""
eval_firebig.py
Compare model predictions with ground truth firebig.
Computes TP, FP, FN, Precision, Recall, F1 (IoU=0.5), and FPS.
"""

import json
import os
import time


# =========================================================
# Configuration (server paths)
# =========================================================

GROUND_TRUTH_JSON = "/home/aistudio/competition/dataset/annotations/val.json"
PREDICTION_JSON  = "/home/aistudio/result.json"

IOU_THRESHOLD = 0.5


# =========================================================
# Helper: compute IoU between two [x, y, w, h] boxes
# =========================================================

def compute_iou(box_a, box_b):
    """box format: [x, y, w, h] (pixels), COCO style"""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


# =========================================================
# Step 1: extract ground truth firebig from COCO val.json
# =========================================================

with open(GROUND_TRUTH_JSON, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

# Build image_id -> file_name lookup
id_to_name = {}
for img in gt_data.get("images", []):
    id_to_name[img["id"]] = img["file_name"]  # e.g. "fire_00003.jpg"

# Group annotations by image_id
gt_by_image = {}
for ann in gt_data.get("annotations", []):
    img_id = ann["image_id"]
    if img_id not in gt_by_image:
        gt_by_image[img_id] = []
    gt_by_image[img_id].append(ann)

# For each image, find the largest bbox -> ground truth firebig
# key: file_name_stem (e.g. "fire_00003"), value: [x, y, w, h]
gt_firebig = {}
for img_id, anns in gt_by_image.items():
    largest = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
    file_name = id_to_name.get(img_id, "img_{}.jpg".format(img_id))
    stem = os.path.splitext(file_name)[0]
    # Convert bbox to float [x, y, w, h]
    bbox = [float(v) for v in largest["bbox"]]
    gt_firebig[stem] = bbox

print("GT  images with firebig: {}".format(len(gt_firebig)))


# =========================================================
# Step 2: read predictions (competition format)
# =========================================================

with open(PREDICTION_JSON, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

pred_entries = pred_data.get("result", [])

# key: image_id, value: [x, y, w, h]
pred_firebig = {}
for entry in pred_entries:
    img_id = entry["image_id"]  # e.g. "fire_00003" or "frame_000006"
    # Normalize: strip extension if present
    img_id = os.path.splitext(img_id)[0]
    bbox = [
        float(entry["x"]),
        float(entry["y"]),
        float(entry["width"]),
        float(entry["height"]),
    ]
    pred_firebig[img_id] = bbox

print("Pred images with detections: {}".format(len(pred_firebig)))


# =========================================================
# Step 3: match predictions to ground truth (greedy IoU)
# =========================================================

# Only consider images that exist in both GT and predictions
common_images = set(gt_firebig.keys()) & set(pred_firebig.keys())
gt_only_images = set(gt_firebig.keys()) - set(pred_firebig.keys())
pred_only_images = set(pred_firebig.keys()) - set(gt_firebig.keys())

if not common_images and not gt_only_images and not pred_only_images:
    print("\nWARNING: No overlap between GT and prediction image names!")
    print("GT sample: {}".format(list(gt_firebig.keys())[:3]))
    print("Pred sample: {}".format(list(pred_firebig.keys())[:3]))
    exit(1)

tp = 0
fp = 0
matched_gt = set()

# For predictions on common images, try to match
for img_id in common_images:
    pred_box = pred_firebig[img_id]
    gt_box = gt_firebig[img_id]
    iou = compute_iou(pred_box, gt_box)
    if iou >= IOU_THRESHOLD:
        tp += 1
        matched_gt.add(img_id)
    else:
        fp += 1

# Predictions on images with no GT -> false positives
fp += len(pred_only_images)

# GT on images with no prediction -> false negatives
fn = len(gt_only_images) + (len(common_images) - len(matched_gt))

# =========================================================
# Step 4: compute metrics
# =========================================================

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# =========================================================
# Print results
# =========================================================

print("=" * 60)
print("Evaluation Results (IoU threshold = {})".format(IOU_THRESHOLD))
print("=" * 60)
print("  GT  images total           : {}".format(len(gt_firebig)))
print("  Pred images total          : {}".format(len(pred_firebig)))
print("  Common images (both sides) : {}".format(len(common_images)))
print("  GT-only  (missed)          : {}".format(len(gt_only_images)))
print("  Pred-only (hallucinated)   : {}".format(len(pred_only_images)))
print("-" * 60)
print("  TP  (correct)              : {}".format(tp))
print("  FP  (wrong detection)      : {}".format(fp))
print("  FN  (missed)               : {}".format(fn))
print("-" * 60)
print("  Precision                  : {:.4f}".format(precision))
print("  Recall                     : {:.4f}".format(recall))
print("  F1 Score                   : {:.4f}".format(f1))
print("=" * 60)

# =========================================================
# FPS estimation
# =========================================================

print()
print("FPS Estimation:")
print("  Competition test set has 4544 images, FPS >= 20 required.")
print("  FPS = 4544 / total_inference_seconds")
print("  Current val set: {} images".format(len(pred_firebig)))
print("  Enter your total inference time on the val set (seconds)")
print("  to estimate the competition FPS.")
