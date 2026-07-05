# -*- coding: utf-8 -*-
"""
比赛推理脚本
调用方式:
python predict.py data.txt result.json
"""

import os
import sys
import json
import time
import yaml
import paddle
import numpy as np

# ==========================
# 路径修复（关键）
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PADDLEDET_DIR = os.path.join(BASE_DIR, "PaddleDetection")
DEPLOY_DIR = os.path.join(PADDLEDET_DIR, "deploy", "python")

sys.path.insert(0, PADDLEDET_DIR)
sys.path.insert(0, DEPLOY_DIR)

from paddle.inference import Config, create_predictor

from PaddleDetection.deploy.python.preprocess import (
    preprocess,
    Resize,
    NormalizeImage,
    Permute,
    PadStride
)

# ==========================
# 参数
# ==========================
MODEL_DIR = os.path.join(BASE_DIR, "model")
THRESHOLD = 0.25


class PredictConfig:
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, "infer_cfg.yml")

        if not os.path.exists(deploy_file):
            raise FileNotFoundError(
                f"infer_cfg.yml not found: {deploy_file}"
            )

        with open(deploy_file, "r", encoding="utf-8") as f:
            yml_conf = yaml.safe_load(f)

        self.arch = yml_conf["arch"]
        self.preprocess_infos = yml_conf["Preprocess"]

    def print_config(self):
        print("Model:", self.arch)


def get_test_images(infer_file):
    """
    官方 data.txt:
    test/Image/frame_00000.jpg
    """

    images = []

    with open(infer_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line = line.replace("\\", "/")

        if os.path.exists(line):
            images.append(line)

    if len(images) == 0:
        raise RuntimeError(
            f"No valid image found in {infer_file}"
        )

    return images


def load_predictor(model_dir):

    model_file = os.path.join(
        model_dir,
        "model.pdmodel"
    )

    params_file = os.path.join(
        model_dir,
        "model.pdiparams"
    )

    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)

    if not os.path.exists(params_file):
        raise FileNotFoundError(params_file)

    config = Config(model_file, params_file)

    # GPU优先
    try:
        if paddle.is_compiled_with_cuda():
            config.enable_use_gpu(1000, 0)
            print("GPU enabled")
        else:
            config.disable_gpu()
            print("CPU mode")
    except:
        config.disable_gpu()
        print("CPU mode")

    config.disable_glog_info()
    config.enable_memory_optim()

    # 稳定性更高
    config.switch_ir_optim(True)

    predictor = create_predictor(config)

    return predictor


def create_inputs(imgs, im_info):

    inputs = {}

    im_shape = []
    scale_factor = []

    for e in im_info:

        im_shape.append(
            np.array(
                (e["im_shape"],)
            ).astype("float32")
        )

        scale_factor.append(
            np.array(
                (e["scale_factor"],)
            ).astype("float32")
        )

    inputs["image"] = np.stack(
        imgs,
        axis=0
    )

    inputs["im_shape"] = np.concatenate(
        im_shape,
        axis=0
    )

    inputs["scale_factor"] = np.concatenate(
        scale_factor,
        axis=0
    )

    return inputs


class Detector:

    def __init__(self, model_dir):

        self.pred_config = PredictConfig(
            model_dir
        )

        self.predictor = load_predictor(
            model_dir
        )

        self.preprocess_ops = self.get_ops()

    def get_ops(self):

        preprocess_ops = []

        for op_info in self.pred_config.preprocess_infos:

            new_op_info = op_info.copy()

            op_type = new_op_info.pop("type")

            preprocess_ops.append(
                eval(op_type)(**new_op_info)
            )

        return preprocess_ops

    def predict(self, inputs):

        input_names = self.predictor.get_input_names()

        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])

        self.predictor.run()
        print(self.predictor.get_output_names())
        output_names = self.predictor.get_output_names()

        outputs = []

        for name in output_names:
            out = self.predictor.get_output_handle(
                name
            ).copy_to_cpu()

            outputs.append(out)

        boxes = None
        boxes_num = None

        for out in outputs:

            # bbox shape: [N, 6]
            if len(out.shape) == 2 and out.shape[1] == 6:
                boxes = out

            # bbox_num shape: [1]
            elif len(out.shape) == 1:
                boxes_num = out

        return boxes, boxes_num

def predict_image(
    detector,
    image_list,
    result_path
):

    results = {"result": []}

    total = len(image_list)

    for idx, im_path in enumerate(image_list):

        try:

            im, im_info = preprocess(
                im_path,
                detector.preprocess_ops
            )

            inputs = create_inputs(
                [im],
                [im_info]
            )

            boxes, boxes_num = (
                detector.predict(inputs)
            )

            image_id = os.path.splitext(
                os.path.basename(im_path)
            )[0]

            det_num = int(
                boxes_num[0]
            )

            if det_num <= 0:
                continue

            det_boxes = boxes[:det_num]

            for box in det_boxes:

                cls_id = int(box[0])
                score = float(box[1])

                if score < THRESHOLD:
                    continue

                xmin = max(
                    0.0,
                    float(box[2])
                )

                ymin = max(
                    0.0,
                    float(box[3])
                )

                xmax = float(box[4])
                ymax = float(box[5])

                width = max(
                    0.0,
                    xmax - xmin
                )

                height = max(
                    0.0,
                    ymax - ymin
                )

                results["result"].append(
                    {
                        "image_id":
                            image_id,

                        "type":
                            cls_id + 1,

                        "x":
                            round(xmin, 2),

                        "y":
                            round(ymin, 2),

                        "width":
                            round(width, 2),

                        "height":
                            round(height, 2),

                        "segmentation":
                            []
                    }
                )

            if idx % 100 == 0:
                print(
                    f"{idx}/{total}"
                )

        except Exception as e:

            print(
                f"Error: {im_path}"
            )

            print(e)

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False
        )

    print(
        "Saved:",
        result_path
    )

    print(
        "Total detections:",
        len(results["result"])
    )


def main():

    if len(sys.argv) != 3:
        print(
            "Usage:"
        )
        print(
            "python predict.py data.txt result.json"
        )
        sys.exit(1)

    infer_txt = sys.argv[1]
    result_path = sys.argv[2]

    paddle.enable_static()

    detector = Detector(
        MODEL_DIR
    )

    image_list = get_test_images(
        infer_txt
    )

    predict_image(
        detector,
        image_list,
        result_path
    )


if __name__ == "__main__":

    start = time.time()

    main()

    print(
        "Total time:",
        round(
            time.time() - start,
            2
        ),
        "s"
    )