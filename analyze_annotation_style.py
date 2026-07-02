
# -*- coding: utf-8 -*-
"""
COCO标注风格分析器
以“补全版”为标准，分析官方漏标规律

作者：ChatGPT
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================
# 路径
# ==========================
MY_LABEL_DIR = r"D:\ruanjiain\A_train\A_train\label"
OFFICIAL_DIR = r"D:\ruanjiain\A_train_out\A_train\label"

SAVE_DIR = "./analysis_result"
os.makedirs(SAVE_DIR, exist_ok=True)

# IoU匹配阈值
IOU_THRESH = 0.6

# 边缘判定（占宽高比例）
EDGE_RATIO = 0.08

# ==========================
# 工具函数
# ==========================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def shape_to_bbox(shape):
    """
    LabelMe rectangle/polygon -> bbox
    """
    pts = np.array(shape['points'])

    x1 = pts[:, 0].min()
    y1 = pts[:, 1].min()
    x2 = pts[:, 0].max()
    y2 = pts[:, 1].max()

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2)
    ]


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union


def parse_labelme(json_path):
    data = load_json(json_path)

    h = data.get("imageHeight", None)
    w = data.get("imageWidth", None)

    objects = []

    for shape in data['shapes']:
        label = shape['label']

        bbox = shape_to_bbox(shape)

        objects.append({
            "label": label,
            "bbox": bbox
        })

    return objects, w, h


def bbox_feature(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    area = bw * bh

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    rel_x = cx / img_w
    rel_y = cy / img_h

    dist_center = np.sqrt(
        (rel_x - 0.5) ** 2 +
        (rel_y - 0.5) ** 2
    )

    near_edge = (
        x1 < img_w * EDGE_RATIO or
        y1 < img_h * EDGE_RATIO or
        x2 > img_w * (1 - EDGE_RATIO) or
        y2 > img_h * (1 - EDGE_RATIO)
    )

    truncation = (
        x1 <= 2 or
        y1 <= 2 or
        x2 >= img_w - 2 or
        y2 >= img_h - 2
    )

    return {
        "width": bw,
        "height": bh,
        "area": area,
        "aspect_ratio": bw / (bh + 1e-6),
        "center_x": rel_x,
        "center_y": rel_y,
        "dist_center": dist_center,
        "near_edge": int(near_edge),
        "truncation": int(truncation)
    }


# ==========================
# 主分析
# ==========================

missing_records = []
official_stats = []
my_stats = []

json_files = [
    f for f in os.listdir(MY_LABEL_DIR)
    if f.endswith(".json")
]

for jf in tqdm(json_files):

    my_json = os.path.join(MY_LABEL_DIR, jf)
    off_json = os.path.join(OFFICIAL_DIR, jf)

    if not os.path.exists(off_json):
        continue

    try:
        my_objs, w, h = parse_labelme(my_json)
        off_objs, _, _ = parse_labelme(off_json)

    except Exception as e:
        print("error:", jf, e)
        continue

    # ----------------------
    # 官方风格统计
    # ----------------------
    for obj in off_objs:
        feat = bbox_feature(
            obj['bbox'],
            w, h
        )
        feat["label"] = obj['label']
        official_stats.append(feat)

    for obj in my_objs:
        feat = bbox_feature(
            obj['bbox'],
            w, h
        )
        feat["label"] = obj['label']
        my_stats.append(feat)

    # ----------------------
    # 漏标分析
    # ----------------------
    for my_obj in my_objs:

        matched = False

        for off_obj in off_objs:

            if my_obj['label'] != off_obj['label']:
                continue

            iou = bbox_iou(
                my_obj['bbox'],
                off_obj['bbox']
            )

            if iou > IOU_THRESH:
                matched = True
                break

        if not matched:

            feat = bbox_feature(
                my_obj['bbox'],
                w, h
            )

            feat["label"] = my_obj['label']
            feat["file"] = jf

            missing_records.append(feat)

# ==========================
# 保存统计
# ==========================

missing_df = pd.DataFrame(missing_records)
official_df = pd.DataFrame(official_stats)
my_df = pd.DataFrame(my_stats)

missing_df.to_csv(
    os.path.join(SAVE_DIR, "missing_objects.csv"),
    index=False,
    encoding='utf-8-sig'
)

# ==========================
# 统计输出
# ==========================

summary = []

summary.append("===== 漏标统计 =====\n")

if missing_df.empty:
    summary.append("无漏标对象")
else:
    for cls in missing_df['label'].unique():

        cls_df = missing_df[
            missing_df['label'] == cls
        ]

        summary.append(f"{cls}")
        summary.append(
            f"漏标数量: {len(cls_df)}"
        )

        summary.append(
            f"平均面积: "
            f"{cls_df['area'].mean():.1f}"
        )

        summary.append(
            f"边缘比例: "
            f"{cls_df['near_edge'].mean():.2%}"
        )

        summary.append(
            f"截断比例: "
            f"{cls_df['truncation'].mean():.2%}"
        )

        summary.append(
            f"中心距离均值: "
            f"{cls_df['dist_center'].mean():.3f}"
        )

        summary.append("\n")

# ==========================
# 官方保留 vs 漏标 风格差异
# ==========================

summary.append("===== 官方保留 vs 漏标 =====\n")

for cls in ['fire', 'board', 'battery']:

    off_cls = official_df[
        official_df['label'] == cls
    ] if not official_df.empty else pd.DataFrame()

    miss_cls = missing_df[
        missing_df['label'] == cls
    ] if not missing_df.empty else pd.DataFrame()

    if len(off_cls) == 0 or len(miss_cls) == 0:
        continue

    summary.append(f"==== {cls}风格差异 ====")

    summary.append(
        f"官方面积均值: "
        f"{off_cls['area'].mean():.1f}"
    )

    summary.append(
        f"漏标面积均值: "
        f"{miss_cls['area'].mean():.1f}"
    )

    summary.append(
        f"官方中心距离: "
        f"{off_cls['dist_center'].mean():.3f}"
    )

    summary.append(
        f"漏标中心距离: "
        f"{miss_cls['dist_center'].mean():.3f}"
    )

    summary.append(
        f"官方边缘比例: "
        f"{off_cls['near_edge'].mean():.2%}"
    )

    summary.append(
        f"漏标边缘比例: "
        f"{miss_cls['near_edge'].mean():.2%}"
    )

    summary.append("\n")

with open(
    os.path.join(SAVE_DIR, "summary.txt"),
    'w',
    encoding='utf-8'
) as f:

    f.write("\n".join(summary))

# ==========================
# 可视化
# ==========================

if not official_df.empty and not my_df.empty:
    plt.figure(figsize=(8, 5))

    plt.hist(
        official_df['area'],
        bins=60,
        alpha=0.5,
        label='official'
    )

    plt.hist(
        my_df['area'],
        bins=60,
        alpha=0.5,
        label='fixed'
    )

    plt.legend()
    plt.xlabel("bbox area")
    plt.ylabel("count")
    plt.title("bbox area distribution")

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "bbox_distribution.png"
        )
    )

    plt.close()
else:
    print("跳过面积分布图: official_df 或 my_df 为空")

# 漏标热力图
if not missing_df.empty:
    plt.figure(figsize=(8, 8))

    plt.scatter(
        missing_df['center_x'],
        missing_df['center_y'],
        alpha=0.3,
        s=5
    )

    plt.gca().invert_yaxis()

    plt.title("Missing Object Position")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "missing_heatmap.png"
        )
    )

    plt.close()
else:
    print("跳过漏标热力图: 无漏标对象")

print("\n分析完成")
print("输出目录:", SAVE_DIR)