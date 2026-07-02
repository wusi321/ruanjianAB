# -*- coding: utf-8 -*-
"""
generate_det_txt_firebig.py
Convert COCO-format model detection JSON to per-image YOLO txt files.
Only keeps the LARGEST fire detection per image (firebig).
Server paths match generate_txt.py.
"""

import json
import os
import sys


# =========================================================
# Configuration (same server paths as generate_txt.py)
# =========================================================

# Input: COCO-format detection JSON from model inference
DETECTION_JSON = "/home/aistudio/result.json"

# Output: directory for per-image YOLO-format txt files
OUTPUT_TXT_DIR = "/home/aistudio/det_txt_firebig"

# Score threshold (keep detections with score > this)
FILTER_SCORE = 0.0

# Image directory (fallback for dimensions if not in JSON)
IMAGE_DIR = "/home/aistudio/competition/dataset/val/images"


# =========================================================
# Core logic
# =========================================================

def convert_coco_to_yolo_txt(json_path, output_dir, filter_score=0.0,
                             image_dir=None):
    """
    Read COCO detection JSON, keep only the largest fire detection
    per image, write YOLO-format txt files.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build image_id -> image info lookup
    img_info = {}
    for img in data.get("images", []):
        img_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img.get("width", 640),
            "height": img.get("height", 480),
        }

    # Group annotations by image_id, filter by score
    anns_by_image = {}
    for ann in data.get("annotations", []):
        score = ann.get("score", 1.0)
        if score <= filter_score:
            continue
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    os.makedirs(output_dir, exist_ok=True)

    # For each image, find the largest detection and write YOLO txt
    written_count = 0
    for img_id, anns in anns_by_image.items():
        # Find largest detection by bbox area
        largest = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])

        # Look up image dimensions
        if img_id in img_info:
            w_img = img_info[img_id]["width"]
            h_img = img_info[img_id]["height"]
            fname = img_info[img_id]["file_name"]
        else:
            w_img = 640
            h_img = 480
            fname = f"img_{img_id}.jpg"

        # COCO bbox: [x, y, w, h] (pixels)
        bx, by, bw, bh = largest["bbox"]

        # Convert to YOLO format: class_id cx cy w h (normalized 0-1)
        cx = (bx + bw / 2.0) / w_img
        cy = (by + bh / 2.0) / h_img
        nw = bw / w_img
        nh = bh / h_img

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        # Write YOLO txt (class_id = 0 for firebig, single class)
        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, "w") as f:
            f.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cx, cy, nw, nh))

        written_count += 1

    print("=" * 50)
    print("Input JSON: {}".format(json_path))
    print("Total images with detections: {}".format(len(anns_by_image)))
    print("Output txt files written: {}".format(written_count))
    print("Output directory: {}".format(output_dir))
    print("=" * 50)


if __name__ == "__main__":
    convert_coco_to_yolo_txt(
        DETECTION_JSON,
        OUTPUT_TXT_DIR,
        filter_score=FILTER_SCORE,
        image_dir=IMAGE_DIR,
    )
