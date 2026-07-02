# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================
# 路径
# ==========================

MY_LABEL_DIR = r"D:\ruanjiain\A_train\label"
OFFICIAL_DIR = r"D:\ruanjiain\A_train_out\A_train\label"

SAVE_DIR = "./analysis_result"
os.makedirs(SAVE_DIR, exist_ok=True)

IOU_THRESH = 0.6

# ==========================
# 基础函数
# ==========================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def shape_to_bbox(shape):
    pts = np.array(shape['points'])

    x1 = pts[:, 0].min()
    y1 = pts[:, 1].min()
    x2 = pts[:, 0].max()
    y2 = pts[:, 1].max()

    return [x1, y1, x2, y2]


def bbox_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (
        (box1[2]-box1[0]) *
        (box1[3]-box1[1])
    )

    area2 = (
        (box2[2]-box2[0]) *
        (box2[3]-box2[1])
    )

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union


def parse_json(json_path):

    data = load_json(json_path)

    h = data["imageHeight"]
    w = data["imageWidth"]

    objs = []

    for shape in data["shapes"]:

        bbox = shape_to_bbox(shape)

        objs.append({
            "label": shape["label"],
            "bbox": bbox
        })

    return objs, w, h


def feature(box, w, h):

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    area = bw * bh

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    rel_x = cx / w
    rel_y = cy / h

    dist = np.sqrt(
        (rel_x - 0.5)**2 +
        (rel_y - 0.5)**2
    )

    return {
        "area": area,
        "dist_center": dist
    }


# ==========================
# 主统计
# ==========================

official_keep = []
official_missing = []

json_files = [
    x for x in os.listdir(MY_LABEL_DIR)
    if x.endswith(".json")
]

for jf in tqdm(json_files):

    my_json = os.path.join(
        MY_LABEL_DIR,
        jf
    )

    off_json = os.path.join(
        OFFICIAL_DIR,
        jf
    )

    if not os.path.exists(off_json):
        continue

    try:
        my_objs, w, h = parse_json(my_json)
        off_objs, _, _ = parse_json(off_json)

    except:
        continue

    matched_idx = set()

    for my_obj in my_objs:

        matched = False

        for idx, off_obj in enumerate(off_objs):

            if idx in matched_idx:
                continue

            if (
                my_obj["label"] !=
                off_obj["label"]
            ):
                continue

            iou = bbox_iou(
                my_obj["bbox"],
                off_obj["bbox"]
            )

            if iou > IOU_THRESH:

                matched = True
                matched_idx.add(idx)

                feat = feature(
                    my_obj["bbox"],
                    w, h
                )

                feat["label"] = my_obj["label"]

                official_keep.append(feat)

                break

        if not matched:

            feat = feature(
                my_obj["bbox"],
                w, h
            )

            feat["label"] = my_obj["label"]

            official_missing.append(feat)

# dataframe
keep_df = pd.DataFrame(official_keep)
miss_df = pd.DataFrame(official_missing)

# ==========================
# 绘图
# ==========================

classes = [
    "fire",
    "board",
    "battery"
]

summary = []

for cls in classes:

    keep_cls = keep_df[
        keep_df.label == cls
    ]

    miss_cls = miss_df[
        miss_df.label == cls
    ]

    if len(keep_cls) == 0:
        continue

    # ----------------------
    # 面积分布
    # ----------------------

    plt.figure(figsize=(8,5))

    plt.hist(
        np.log10(
            keep_cls.area + 1
        ),
        bins=40,
        alpha=0.5,
        label="official_keep"
    )

    plt.hist(
        np.log10(
            miss_cls.area + 1
        ),
        bins=40,
        alpha=0.5,
        label="official_missing"
    )

    plt.legend()

    plt.xlabel(
        "log10(bbox area)"
    )

    plt.ylabel("count")

    plt.title(
        f"{cls} area distribution"
    )

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            f"{cls}_distribution.png"
        )
    )

    plt.close()

    # ----------------------
    # 中心距离分布
    # ----------------------

    plt.figure(figsize=(8,5))

    plt.hist(
        keep_cls.dist_center,
        bins=40,
        alpha=0.5,
        label="official_keep"
    )

    plt.hist(
        miss_cls.dist_center,
        bins=40,
        alpha=0.5,
        label="official_missing"
    )

    plt.legend()

    plt.xlabel(
        "distance to center"
    )

    plt.ylabel("count")

    plt.title(
        f"{cls} center distribution"
    )

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            f"{cls}_center_distribution.png"
        )
    )

    plt.close()

    # ----------------------
    # 小中大目标比例
    # ----------------------

    q1 = keep_cls.area.quantile(0.33)
    q2 = keep_cls.area.quantile(0.66)

    def size_ratio(df):

        small = (
            df.area < q1
        ).mean()

        medium = (
            (df.area >= q1) &
            (df.area < q2)
        ).mean()

        large = (
            df.area >= q2
        ).mean()

        return (
            small,
            medium,
            large
        )

    k_s, k_m, k_l = size_ratio(
        keep_cls
    )

    m_s, m_m, m_l = size_ratio(
        miss_cls
    )

    summary.append(
        f"===== {cls} ====="
    )

    summary.append(
        f"官方保留数量: {len(keep_cls)}"
    )

    summary.append(
        f"官方漏标数量: {len(miss_cls)}"
    )

    summary.append(
        f"保留 小/中/大: "
        f"{k_s:.1%}/"
        f"{k_m:.1%}/"
        f"{k_l:.1%}"
    )

    summary.append(
        f"漏标 小/中/大: "
        f"{m_s:.1%}/"
        f"{m_m:.1%}/"
        f"{m_l:.1%}"
    )

    summary.append("\n")

with open(
    os.path.join(
        SAVE_DIR,
        "compare_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )

print("分析完成")
print(SAVE_DIR)