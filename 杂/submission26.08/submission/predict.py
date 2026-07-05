import os
import sys
import json
import time
import cv2
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor


# =========================================================
# 路径
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")


# =========================================================
# 参数
# =========================================================

INPUT_SIZE = 640

# 提高阈值减少重复框
SCORE_THRESH = 0.03

# 类别映射
CLASS_MAP = {
    0: 1,   # battery
    1: 2,   # board
    2: 3    # fire
}

def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def nms(results, iou_thresh=0.5):

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    keep = []

    while results:

        best = results.pop(0)

        keep.append(best)

        remain = []

        for item in results:

            if item["type"] != best["type"]:
                remain.append(item)
                continue

            box1 = [
                best["x"],
                best["y"],
                best["x"] + best["width"],
                best["y"] + best["height"]
            ]

            box2 = [
                item["x"],
                item["y"],
                item["x"] + item["width"],
                item["y"] + item["height"]
            ]

            if iou(box1, box2) < iou_thresh:
                remain.append(item)

        results = remain

    return keep
# =========================================================
# 创建 Predictor
# =========================================================

def load_predictor():

    model_file = os.path.join(
        MODEL_DIR,
        "model.pdmodel"
    )

    params_file = os.path.join(
        MODEL_DIR,
        "model.pdiparams"
    )

    config = Config(
        model_file,
        params_file
    )

    # GPU
    config.enable_use_gpu(2048, 0)

    # IR优化
    config.switch_ir_optim(True)

    # 内存优化
    config.enable_memory_optim()

    # ZeroCopy
    config.switch_use_feed_fetch_ops(True)

    # 不输出GLOG日志
    config.disable_glog_info()

    predictor = create_predictor(config)

    return predictor


# =========================================================
# 预处理
# =========================================================

# =========================================================
# 预处理
# =========================================================



# def preprocess(img):

#     raw_h, raw_w = img.shape[:2]

#     # 直接缩放
#     img_resized = cv2.resize(
#         img,
#         (INPUT_SIZE, INPUT_SIZE)
#     )

#     img_resized = img_resized.astype(np.float32)

#     img_resized = img_resized / 255.0

#     mean = np.array(
#         [0.485, 0.456, 0.406],
#         dtype=np.float32
#     )

#     std = np.array(
#         [0.229, 0.224, 0.225],
#         dtype=np.float32
#     )

#     img_resized = (img_resized - mean) / std

#     # HWC -> CHW
#     img_resized = img_resized.transpose((2, 0, 1))

#     # BCHW
#     img_resized = np.expand_dims(
#         img_resized,
#         axis=0
#     ).astype(np.float32)

#     # 缩放比例
#     scale_y = float(INPUT_SIZE) / float(raw_h)
#     scale_x = float(INPUT_SIZE) / float(raw_w)

#     return (
#         img_resized,
#         scale_x,
#         scale_y,
#         raw_w,
#         raw_h
#     )
def preprocess(img):

    raw_h, raw_w = img.shape[:2]

    img_resized = cv2.resize(
        img,
        (INPUT_SIZE, INPUT_SIZE)
    )

    img_resized = img_resized.astype(np.float32)

    img_resized = img_resized / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img_resized = (img_resized - mean) / std

    img_resized = img_resized.transpose((2, 0, 1))

    img_resized = np.expand_dims(
        img_resized,
        axis=0
    ).astype(np.float32)

    scale_y = float(INPUT_SIZE) / float(raw_h)
    scale_x = float(INPUT_SIZE) / float(raw_w)

    return (
        img_resized,
        scale_x,
        scale_y,
        raw_w,
        raw_h
    )
# def preprocess(img):

#     raw_h, raw_w = img.shape[:2]

#     img = cv2.resize(
#         img,
#         (INPUT_SIZE, INPUT_SIZE)
#     )

#     img = img.astype(np.float32)

#     # 只归一化
#     img = img / 255.0

#     # HWC -> CHW
#     img = img.transpose((2, 0, 1))

#     # BCHW
#     img = np.expand_dims(
#         img,
#         axis=0
#     )

#     scale_y = INPUT_SIZE / raw_h
#     scale_x = INPUT_SIZE / raw_w

#     return (
#         img.astype(np.float32),
#         scale_x,
#         scale_y,
#         raw_w,
#         raw_h
#     )
# =========================================================
# 推理单张图片
# =========================================================

# =========================================================
# 推理单张图片
# =========================================================

def predict_image(
    predictor,
    image_path
):

    img = cv2.imread(image_path)

    if img is None:
        return []

    # input_data, scale_x, scale_y, raw_w, raw_h = preprocess(img)
    # (
    #     input_data,
    #     scale_factor,
    #     raw_w,
    #     raw_h,
    #     pad_x,
    #     pad_y,
    #     scale
    # ) = preprocess(img)
    (
        input_data,
        scale_x,
        scale_y,
        raw_w,
        raw_h
    ) = preprocess(img)
    # =====================================================
    # PaddleDetection 输入
    # =====================================================

    scale_factor = np.array(
        [[scale_y, scale_x]],
        dtype=np.float32
    )

    # im_shape = np.array(
    #     [[INPUT_SIZE, INPUT_SIZE]],
    #     dtype=np.float32
    # )

    input_names = predictor.get_input_names()

    # for name in input_names:

    #     input_tensor = predictor.get_input_handle(name)

    #     # image
    #     if "image" in name:
    #         input_tensor.copy_from_cpu(input_data)

    #     # scale_factor
    #     elif "scale_factor" in name:
    #         input_tensor.copy_from_cpu(scale_factor)

    #     # im_shape
    #     elif "im_shape" in name:
    #         input_tensor.copy_from_cpu(im_shape)
    for name in input_names:

        input_tensor = predictor.get_input_handle(name)

        print(name)

        if name == "x":
            input_tensor.copy_from_cpu(input_data)

        elif name == "image":
            input_tensor.copy_from_cpu(input_data)

        elif name == "scale_factor":
            input_tensor.copy_from_cpu(scale_factor)

        # elif name == "im_shape":
        #     input_tensor.copy_from_cpu(im_shape)
    # =====================================================
    # 推理
    # =====================================================

    predictor.run()

    # =====================================================
    # 获取输出
    # =====================================================

    output_names = predictor.get_output_names()

    print("OUTPUT NAMES:", output_names)

    boxes = None

    for name in output_names:

        output_tensor = predictor.get_output_handle(name)

        output_data = output_tensor.copy_to_cpu()

        print(name, output_data.shape)

        # 检测框输出
        if len(output_data.shape) == 2:

            # 6列:
            # [class_id, score, xmin, ymin, xmax, ymax]

            if output_data.shape[1] == 6:
                boxes = output_data

            # 7列:
            # [batch_id, class_id, score, xmin, ymin, xmax, ymax]

            elif output_data.shape[1] == 7:
                boxes = output_data

    if boxes is None:
        return []

    # =====================================================
    # image_id
    # =====================================================

    image_id = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # =====================================================
    # 保存结果
    # =====================================================

    results = []

    # =====================================================
    # 解析输出
    # =====================================================

    for item in boxes:
        #print(item)
        item = item.tolist()

        # -------------------------------------------------
        # 6列输出
        # [class_id, score, xmin, ymin, xmax, ymax]
        # -------------------------------------------------

        if len(item) == 6:

            clsid = int(item[0])

            score = float(item[1])

            xmin = float(item[2])
            ymin = float(item[3])
            xmax = float(item[4])
            ymax = float(item[5])

        # -------------------------------------------------
        # 7列输出
        # [batch_id, class_id, score, xmin, ymin, xmax, ymax]
        # -------------------------------------------------

        elif len(item) == 7:

            batch_id = int(item[0])

            if batch_id < 0:
                continue

            clsid = int(item[1])

            score = float(item[2])

            xmin = float(item[3])
            ymin = float(item[4])
            xmax = float(item[5])
            ymax = float(item[6])

        else:
            continue

        # =================================================
        # 分数过滤
        # =================================================

        if score < SCORE_THRESH:
            continue

        # =================================================
        # 类别过滤
        # =================================================

        if clsid not in CLASS_MAP:
            continue

        # =================================================
        # 边界裁剪
        # =================================================
        # xmin = (xmin - pad_x) / scale
        # ymin = (ymin - pad_y) / scale
        # xmax = (xmax - pad_x) / scale
        # ymax = (ymax - pad_y) / scale
        # xmin = xmin / scale_x
        # ymin = ymin / scale_y
        # xmax = xmax / scale_x
        # ymax = ymax / scale_y
        xmin = max(0, xmin)
        ymin = max(0, ymin)

        xmax = min(raw_w - 1, xmax)
        ymax = min(raw_h - 1, ymax)

        width = xmax - xmin
        height = ymax - ymin

        # =================================================
        # 异常框过滤
        # =================================================

        if width <= 2 or height <= 2:
            continue

        if width > raw_w * 0.95:
            continue

        if height > raw_h * 0.95:
            continue
        # 超大框过滤
        if width * height > raw_w * raw_h * 0.5:
            continue
        # =================================================
        # 保存结果
        # =================================================

        # results.append({

        #     "image_id": image_id,

        #     "type": CLASS_MAP[clsid],

        #     "x": round(float(xmin), 2),

        #     "y": round(float(ymin), 2),

        #     "width": round(float(width), 2),

        #     "height": round(float(height), 2),

        #     "segmentation": []

        # })
        results.append({

            "image_id": image_id,

            "type": CLASS_MAP[clsid],

            "x": round(float(xmin), 2),

            "y": round(float(ymin), 2),

            "width": round(float(width), 2),

            "height": round(float(height), 2),

            "score": float(score),

            "segmentation": []

        })
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    #results = results[:10]
    results = nms(results, 0.3)
    return results
# =========================================================
# 主函数
# =========================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage: python predict.py data.txt result.json"
        )

        return

    txt_path = sys.argv[1]

    save_path = sys.argv[2]

    # 读取图片列表
    with open(txt_path, "r") as f:

        image_paths = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    print(f"Total Images: {len(image_paths)}")

    predictor = load_predictor()

    final_results = []

    start_time = time.time()

    for idx, image_path in enumerate(image_paths):

        try:

            result = predict_image(
                predictor,
                image_path
            )

            final_results.extend(result)

        except Exception as e:

            print(f"ERROR: {image_path}")
            print(e)

        if (idx + 1) % 50 == 0:

            print(
                f"Processed: "
                f"{idx + 1}/{len(image_paths)}"
            )

    total_time = time.time() - start_time

    fps = len(image_paths) / total_time

    print("\n========================")
    print(f"FPS: {fps:.2f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Detections: {len(final_results)}")
    print("========================")

    # 保存json
    output_json = {
        "result": final_results
    }

    with open(save_path, "w") as f:

        json.dump(
            output_json,
            f,
            ensure_ascii=False
        )

    print(f"Saved: {save_path}")


# =========================================================
# 入口
# =========================================================

if __name__ == "__main__":

    main()