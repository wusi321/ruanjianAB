#```python
# -*- coding: utf-8 -*-
"""
可视化 LabelMe 检测框

功能：
1. 遍历所有图片和 json
2. 画出 bbox + 类别名
3. 输出到 vis_labels 文件夹
4. 统计空标、异常框
5. 方便人工快速检查漏标/错标

目录结构:
A_train/
├── Image/
├── label/
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# ==========================
# 路径配置
# ==========================
#ROOT_DIR = "A_train_out\A_train"
ROOT_DIR = "A_train"
IMG_DIR = os.path.join(ROOT_DIR, "Image")
LABEL_DIR = os.path.join(ROOT_DIR, "label")

SAVE_DIR = os.path.join(ROOT_DIR, "vis_labels2")
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================
# 类别颜色（BGR）
# ==========================
COLORS = {
    "battery": (0, 255, 0),   # 绿
    "board": (255, 0, 0),     # 蓝
    "fire": (0, 0, 255),      # 红
}

# ==========================
# 字体
# ==========================
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_label(img, text, x1, y1, color):
    """
    绘制类别标签背景
    """
    font_scale = 0.7
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        text,
        FONT,
        font_scale,
        thickness
    )

    cv2.rectangle(
        img,
        (x1, y1 - th - 10),
        (x1 + tw + 8, y1),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x1 + 4, y1 - 5),
        FONT,
        font_scale,
        (255, 255, 255),
        thickness
    )


def parse_points(points):
    """
    LabelMe points -> bbox
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(xs))
    y2 = int(max(ys))

    return x1, y1, x2, y2


def main():

    image_files = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(".jpg")
    ])

    total_boxes = 0
    empty_count = 0
    bad_box_count = 0
    missing_json = []

    for img_name in tqdm(image_files):

        img_path = os.path.join(
            IMG_DIR,
            img_name
        )

        json_name = (
            os.path.splitext(img_name)[0]
            + ".json"
        )

        json_path = os.path.join(
            LABEL_DIR,
            json_name
        )

        img = cv2.imread(img_path)

        if img is None:
            print(f"读取失败: {img_path}")
            continue

        # ----------------------
        # json不存在
        # ----------------------
        if not os.path.exists(json_path):
            missing_json.append(img_name)

            save_path = os.path.join(
                SAVE_DIR,
                img_name
            )

            cv2.putText(
                img,
                "NO JSON",
                (50, 80),
                FONT,
                2,
                (0, 0, 255),
                3
            )

            cv2.imwrite(save_path, img)
            continue

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        shapes = data.get("shapes", [])

        # ----------------------
        # 空标注
        # ----------------------
        if len(shapes) == 0:
            empty_count += 1

            cv2.putText(
                img,
                "EMPTY LABEL",
                (50, 80),
                FONT,
                2,
                (0, 255, 255),
                3
            )

        # ----------------------
        # 画框
        # ----------------------
        for shape in shapes:

            label = shape.get(
                "label",
                "unknown"
            )

            points = shape.get(
                "points",
                []
            )

            if len(points) < 2:
                bad_box_count += 1
                continue

            x1, y1, x2, y2 = parse_points(
                points
            )

            w = x2 - x1
            h = y2 - y1

            # 异常框
            if w <= 1 or h <= 1:
                bad_box_count += 1
                continue

            total_boxes += 1

            color = COLORS.get(
                label,
                (255, 255, 255)
            )

            # bbox
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # label
            text = (
                f"{label}"
                f" [{w}x{h}]"
            )

            draw_label(
                img,
                text,
                x1,
                y1,
                color
            )

        save_path = os.path.join(
            SAVE_DIR,
            img_name
        )

        cv2.imwrite(
            save_path,
            img
        )

    # ==========================
    # 统计信息
    # ==========================
    print("\n========== 数据集统计 ==========")
    print(f"总图片数: {len(image_files)}")
    print(f"总框数: {total_boxes}")
    print(f"空标图片: {empty_count}")
    print(f"异常框数: {bad_box_count}")
    print(f"缺失JSON: {len(missing_json)}")

    if len(missing_json) > 0:
        print("\n缺失 JSON 文件:")
        for x in missing_json:
            print(x)

    print("\n可视化完成")
    print(f"结果保存在: {SAVE_DIR}")


if __name__ == "__main__":
    main()
#```
