
# -*- coding: utf-8 -*-
"""
3类目标检测推理脚本 (battery / board / fire)
PP-YOLOE / YOLO PaddleDetection 稳定竞赛版
"""

import os
import sys
import json
import time
import numpy as np
import paddle
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, "PaddleDetection"))
sys.path.append(os.path.join(BASE_DIR, "PaddleDetection", "deploy", "python"))

from paddle.inference import Config, create_predictor

MODEL_DIR = os.path.join(BASE_DIR, "model")

THRESHOLD = 0.03


# =========================
# Predictor
# =========================
def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, "model.pdmodel"),
        os.path.join(model_dir, "model.pdiparams")
    )

    try:
        config.enable_use_gpu(1000, 0)
        print("GPU enabled")
    except Exception:
        config.disable_gpu()
        print("CPU mode")

    config.disable_glog_info()
    config.enable_memory_optim()

    # 🔥 必须关闭，否则 conv2d_fusion 容易炸
    config.switch_ir_optim(False)

    config.switch_use_feed_fetch_ops(False)

    return create_predictor(config)


# =========================
# image loader（关键修复点）
# =========================
def load_image(img_path):
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize
    img = cv2.resize(img, (640, 640))

    h1, w1 = 640, 640

    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)

    # 🔥 正确 scale_factor（关键）
    scale_factor = np.array([
        h1 / h0,
        w1 / w0
    ], dtype=np.float32)

    im_shape = np.array([h1, w1], dtype=np.float32)

    return img, im_shape, scale_factor


# =========================
# Detector
# =========================
class Detector:
    def __init__(self, model_dir):
        self.predictor = load_predictor(model_dir)

    def predict(self, img_path):
        img, im_shape, scale_factor = load_image(img_path)

        inputs = {
            "image": img[np.newaxis, :].astype(np.float32),
            "im_shape": im_shape[np.newaxis, :],
            "scale_factor": scale_factor[np.newaxis, :]
        }

        for name in self.predictor.get_input_names():
            self.predictor.get_input_handle(name).copy_from_cpu(inputs[name])

        self.predictor.run()

        outputs = [
            self.predictor.get_output_handle(n).copy_to_cpu()
            for n in self.predictor.get_output_names()
        ]

        boxes = None
        for out in outputs:
            if len(out.shape) == 2 and out.shape[1] == 6:
                boxes = out.astype(np.float32)
                break

        if boxes is None:
            return np.empty((0, 6), dtype=np.float32)

        return boxes


# =========================
# parse
# =========================
def parse_box(box):
    box = np.array(box, dtype=np.float32)

    # 模式1：[x1,y1,x2,y2,score,cls]
    if np.max(box[:4]) > 20 and box[4] <= 1.0:
        x1, y1, x2, y2, score, cls_id = box
    else:
        # 模式2：[cls,score,x1,y1,x2,y2]
        cls_id, score, x1, y1, x2, y2 = box

    cls_id = int(cls_id) % 3  # 强制3类

    return cls_id, float(score), x1, y1, x2, y2


def dedup_boxes(results, iou_thr=0.7):
    # 简化版：按 (cls,x1,y1,x2,y2) hash 去重
    seen = set()
    filtered = []

    for r in results:
        key = (
            r["type"],
            round(r["x"], 1),
            round(r["y"], 1),
            round(r["width"], 1),
            round(r["height"], 1),
        )
        if key in seen:
            continue
        seen.add(key)
        filtered.append(r)

    return filtered



def nms_postfilter(results, iou_thr=0.6):
    final = []

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)

        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        return inter / (area_a + area_b - inter + 1e-6)

    for r in results:
        keep = True
        for fr in final:
            if r["type"] == fr["type"]:
                a = [r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]]
                b = [fr["x"], fr["y"], fr["x"] + fr["width"], fr["y"] + fr["height"]]

                if iou(a, b) > iou_thr:
                    keep = False
                    break

        if keep:
            final.append(r)

    return final
# =========================
# inference
# =========================
def predict_image(detector, image_list, result_path):
    results = {"result": []}

    for idx, im_path in enumerate(image_list):

        image_id = os.path.splitext(os.path.basename(im_path))[0]

        boxes = detector.predict(im_path)

        if len(boxes) == 0:
            continue

        for box in boxes:
            cls_id, score, x1, y1, x2, y2 = parse_box(box)

            if score < THRESHOLD:
                continue

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1, x2), max(y1, y2)

            w = x2 - x1
            h = y2 - y1

            if w <= 1 or h <= 1:
                continue
        
            results["result"].append({
                "image_id": image_id,
                "type": cls_id,
                "x": round(float(x1), 2),
                "y": round(float(y1), 2),
                "width": round(float(w), 2),
                "height": round(float(h), 2),
                "segmentation": []
            })

        if idx % 50 == 0:
            print(f"Progress: {idx}/{len(image_list)}")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print("\nSaved:", result_path)
    print("Total detections:", len(results["result"]))


# =========================
# main
# =========================
def get_images(txt):
    base = os.path.dirname(txt)
    imgs = []

    with open(txt, "r") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.join(base, p)
            imgs.append(p)

    return imgs


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py data.txt result.json")
        return

    paddle.enable_static()

    txt = sys.argv[1]
    out = sys.argv[2]

    detector = Detector(MODEL_DIR)

    imgs = get_images(txt)
    print("Total images:", len(imgs))

    predict_image(detector, imgs, out)


if __name__ == "__main__":
    start = time.time()
    main()
    print("Total time:", round(time.time() - start, 2), "s")