# -*- coding: utf-8 -*-
"""
比赛推理脚本
调用:
python predict.py data.txt result.json
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import paddle

# PaddleDetection 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(
    os.path.join(BASE_DIR, "PaddleDetection")
)

sys.path.append(
    os.path.join(
        BASE_DIR,
        "PaddleDetection",
        "deploy",
        "python"
    )
)

from paddle.inference import (
    Config,
    create_predictor
)

from PaddleDetection.deploy.python.preprocess import (
    preprocess,
    Resize,
    NormalizeImage,
    Permute,
    PadStride
)

# ======================
# 路径
# ======================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

# ======================
# 分类别阈值
# 0:battery
# 1:board
# 2:fire
# ======================

CLASS_THRESHOLDS = {
    0: 0.10,
    1: 0.12,
    2: 0.08
}


class PredictConfig:

    def __init__(self, model_dir):

        deploy_file = os.path.join(
            model_dir,
            "infer_cfg.yml"
        )

        with open(
            deploy_file,
            "r",
            encoding="utf-8"
        ) as f:

            yml_conf = yaml.safe_load(f)

        self.arch = yml_conf["arch"]
        self.preprocess_infos = (
            yml_conf["Preprocess"]
        )


def load_predictor(model_dir):

    model_file = os.path.join(
        model_dir,
        "model.pdmodel"
    )

    params_file = os.path.join(
        model_dir,
        "model.pdiparams"
    )

    config = Config(
        model_file,
        params_file
    )

    # GPU 优先
    try:
        config.enable_use_gpu(
            1000,
            0
        )
        print("GPU enabled")

    except Exception:
        config.disable_gpu()

        # CPU优化
        config.set_cpu_math_library_num_threads(4)

        try:
            config.enable_mkldnn()
        except:
            pass

        print("CPU mode")

    config.disable_glog_info()

    # 内存优化
    config.enable_memory_optim()

    # IR优化（加速）
    config.switch_ir_optim(True)

    config.switch_use_feed_fetch_ops(
        False
    )

    predictor = create_predictor(
        config
    )

    return predictor


def get_test_images(data_txt):

    infer_dir = os.path.dirname(
        os.path.abspath(data_txt)
    )

    image_list = []

    with open(
        data_txt,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        line = line.replace(
            "\\",
            "/"
        )

        if not os.path.isabs(line):

            line = os.path.join(
                infer_dir,
                line
            )

        if os.path.exists(line):
            image_list.append(line)

    return image_list


class Detector:

    def __init__(self, model_dir):

        self.pred_config = (
            PredictConfig(model_dir)
        )

        self.predictor = (
            load_predictor(model_dir)
        )

        self.preprocess_ops = (
            self.get_ops()
        )

    def get_ops(self):

        preprocess_ops = []

        for op_info in (
            self.pred_config
            .preprocess_infos
        ):

            op = op_info.copy()

            op_type = op.pop(
                "type"
            )

            preprocess_ops.append(
                eval(op_type)(**op)
            )

        return preprocess_ops

    def create_inputs(
        self,
        imgs,
        im_info
    ):

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

        inputs["im_shape"] = (
            np.concatenate(
                im_shape,
                axis=0
            )
        )

        inputs["scale_factor"] = (
            np.concatenate(
                scale_factor,
                axis=0
            )
        )

        return inputs

    def predict(self, img_path):

        img, im_info = preprocess(
            img_path,
            self.preprocess_ops
        )

        inputs = self.create_inputs(
            [img],
            [im_info]
        )

        input_names = (
            self.predictor
            .get_input_names()
        )

        for name in input_names:

            input_tensor = (
                self.predictor
                .get_input_handle(name)
            )

            input_tensor.copy_from_cpu(
                inputs[name]
            )

        self.predictor.run()

        output_names = (
            self.predictor
            .get_output_names()
        )

        outputs = []

        for name in output_names:

            out = (
                self.predictor
                .get_output_handle(name)
                .copy_to_cpu()
            )

            outputs.append(out)

        boxes = None

        for out in outputs:

            if (
                len(out.shape) == 2
                and out.shape[1] == 6
            ):
                boxes = out
                break

        if boxes is None:
            return np.empty((0, 6))

        return boxes


def main():

    if len(sys.argv) != 3:

        print(
            "Usage: python predict.py data.txt result.json"
        )

        sys.exit(1)

    data_txt = sys.argv[1]

    result_json = sys.argv[2]

    paddle.enable_static()

    detector = Detector(
        MODEL_DIR
    )

    image_list = get_test_images(
        data_txt
    )

    results = {
        "result": []
    }

    total = len(image_list)

    for idx, img_path in enumerate(
        image_list
    ):

        print(
            f"{idx}/{total}"
        )

        image_id = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        boxes = detector.predict(
            img_path
        )

        for box in boxes:

            cls_id = int(box[0])

            score = float(box[1])

            threshold = (
                CLASS_THRESHOLDS.get(
                    cls_id,
                    0.10
                )
            )

            if score < threshold:
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

            results["result"].append({

                "image_id":
                    image_id,

                "type":
                    cls_id + 1,

                "x":
                    round(
                        xmin,
                        2
                    ),

                "y":
                    round(
                        ymin,
                        2
                    ),

                "width":
                    round(
                        width,
                        2
                    ),

                "height":
                    round(
                        height,
                        2
                    ),

                "segmentation":
                    []
            })

    with open(
        result_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f
        )

    print(
        "Saved:",
        result_json
    )

    print(
        "Total detections:",
        len(
            results["result"]
        )
    )


if __name__ == "__main__":

    start = time.time()

    main()

    print(
        "Total time:",
        round(
            time.time() - start,
            3
        )
    )