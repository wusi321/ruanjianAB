# -*- coding: utf-8 -*-
"""
3类目标检测推理脚本 (battery / board / fire)
YOLO / PP-YOLOE PaddleDetection 推理稳定版

调用方式:
    python predict.py data.txt result.json
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import paddle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, "PaddleDetection"))
sys.path.append(os.path.join(BASE_DIR, "PaddleDetection", "deploy", "python"))

from paddle.inference import Config, create_predictor
from PaddleDetection.deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride


MODEL_DIR = os.path.join(BASE_DIR, "model")

# ⚠️ 竞赛推荐：先低后高
THRESHOLD = 0.01


class PredictConfig:
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, "infer_cfg.yml")
        with open(deploy_file, "r", encoding="utf-8") as f:
            yml_conf = yaml.safe_load(f)

        self.arch = yml_conf["arch"]
        self.preprocess_infos = yml_conf["Preprocess"]

        print("\nModel Arch:", self.arch)
        for op_info in self.preprocess_infos:
            print("  Preprocess op:", op_info["type"])


def load_predictor(model_dir):
    model_file = os.path.join(model_dir, "model.pdmodel")
    params_file = os.path.join(model_dir, "model.pdiparams")

    config = Config(model_file, params_file)

    try:
        config.enable_use_gpu(1000, 0)
        print("GPU enabled")
    except Exception:
        config.disable_gpu()
        print("CPU mode")

    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    config.switch_use_feed_fetch_ops(False)

    return create_predictor(config)


def get_test_images(data_txt):
    infer_dir = os.path.dirname(os.path.abspath(data_txt))
    image_list = []

    with open(data_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace("\\", "/")

            if not os.path.isabs(line):
                line = os.path.join(infer_dir, line)

            if os.path.exists(line):
                image_list.append(line)

    assert len(image_list) > 0, "no image found in {}".format(data_txt)
    return image_list


class Detector:
    def __init__(self, model_dir):
        self.pred_config = PredictConfig(model_dir)
        self.predictor = load_predictor(model_dir)
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        ops = []
        for op_info in self.pred_config.preprocess_infos:
            op = op_info.copy()
            op_type = op.pop("type")
            ops.append(eval(op_type)(**op))
        return ops

    def create_inputs(self, imgs, im_info):
        inputs = {}

        im_shape = []
        scale_factor = []

        for e in im_info:
            im_shape.append(np.array(e["im_shape"], dtype=np.float32))
            scale_factor.append(np.array(e["scale_factor"], dtype=np.float32))

        inputs["image"] = np.stack(imgs, axis=0)
        inputs["im_shape"] = np.stack(im_shape, axis=0)
        inputs["scale_factor"] = np.stack(scale_factor, axis=0)

        return inputs

    def predict(self, img_path):
        img, im_info = preprocess(img_path, self.preprocess_ops)
        inputs = self.create_inputs([img], [im_info])

        for name in self.predictor.get_input_names():
            handle = self.predictor.get_input_handle(name)
            handle.copy_from_cpu(inputs[name])

        self.predictor.run()

        outputs = []
        for name in self.predictor.get_output_names():
            out = self.predictor.get_output_handle(name).copy_to_cpu()
            outputs.append(out)

        # =========================
        # 🔥 自动识别 YOLOE / PP-YOLOE 输出
        # =========================
        boxes = None

        for i, out in enumerate(outputs):
            if len(out.shape) == 2 and out.shape[1] == 6:
                boxes = out.copy()
                print(f"[DEBUG] using output[{i}] shape={out.shape}")
                break

        if boxes is None:
            return np.empty((0, 6), dtype=np.float32)

        if len(boxes) > 0:
            print("[DEBUG] sample box:", boxes[0])

        return boxes


def predict_image(detector, image_list, result_path):
    results = {"result": []}

    total = len(image_list)

    for idx, im_path in enumerate(image_list):
        try:
            image_id = os.path.splitext(os.path.basename(im_path))[0]

            boxes = detector.predict(im_path)

            # score debug
            if len(boxes) > 0:
                scores = boxes[:, 1]
                print(f"[DEBUG] max score: {scores.max():.4f}")

                for box in boxes:
                    if len(box) != 6:
                        continue

                    b0, b1, b2, b3, b4, b5 = box

                    # ============================
                    # 自动判断格式
                    # ============================

                    # case A: [cls, score, x1, y1, x2, y2]
                    if b0 < 10 and b1 <= 1.0:
                        cls_id = int(b0)
                        score = float(b1)
                        x1, y1, x2, y2 = b2, b3, b4, b5

                    # case B: [x1, y1, x2, y2, score, cls]
                    else:
                        x1, y1, x2, y2 = b0, b1, b2, b3
                        score = float(b4)
                        cls_id = int(b5)

                    if score < THRESHOLD:
                        continue

                    x1 = max(0.0, float(x1))
                    y1 = max(0.0, float(y1))
                    x2 = max(x1, float(x2))
                    y2 = max(y1, float(y2))

                    w = x2 - x1
                    h = y2 - y1

                    if w <= 1 or h <= 1:
                        continue

                    results["result"].append({
                        "image_id": image_id,
                        "type": cls_id,
                        "x": round(x1, 2),
                        "y": round(y1, 2),
                        "width": round(w, 2),
                        "height": round(h, 2),
                        "segmentation": []
                    })

        except Exception as e:
            print("Error:", im_path, e)

        if idx % 50 == 0:
            print(f"Progress: {idx}/{total}")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print("\nSaved:", result_path)
    print("Total detections:", len(results["result"]))


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py data.txt result.json")
        sys.exit(1)

    paddle.enable_static()

    data_txt = sys.argv[1]
    result_json = sys.argv[2]

    detector = Detector(MODEL_DIR)

    image_list = get_test_images(data_txt)
    print("Total images:", len(image_list))

    predict_image(detector, image_list, result_json)


if __name__ == "__main__":
    start = time.time()
    main()
    print("Total time:", round(time.time() - start, 2), "s")