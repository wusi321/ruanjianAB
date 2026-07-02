
#```python
# -*- coding: utf-8 -*-

import os
import json
from collections import defaultdict
from tqdm import tqdm

# ==========================
# 路径
# ==========================
LABEL_DIR = r"D:\ruanjiain\A_train_out\A_train\label"
#LABEL_DIR = r"D:\ruanjiain\A_train\label"
# ==========================
# 初始化统计
# ==========================
class_count = defaultdict(int)
image_count = defaultdict(int)

total_boxes = 0
empty_images = 0
bad_json = 0

# 小目标统计（面积小于阈值）
small_object_count = defaultdict(int)
SMALL_AREA = 80 * 80

json_files = [
    f for f in os.listdir(LABEL_DIR)
    if f.endswith(".json")
]

print(f"发现 {len(json_files)} 个 json 文件\n")

for jf in tqdm(json_files):

    path = os.path.join(LABEL_DIR, jf)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print(f"读取失败: {jf}")
        print(e)
        bad_json += 1
        continue

    shapes = data.get("shapes", [])

    # 空标
    if len(shapes) == 0:
        empty_images += 1
        continue

    current_image_classes = set()

    for shape in shapes:

        label = shape.get(
            "label",
            "unknown"
        ).strip().lower()

        points = shape.get(
            "points",
            []
        )

        # ----------------------
        # bbox解析（兼容2点/4点）
        # ----------------------
        if len(points) >= 2:

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            w = max(xs) - min(xs)
            h = max(ys) - min(ys)

            area = w * h

            # 小目标统计
            if area < SMALL_AREA:
                small_object_count[label] += 1

        # ----------------------
        # 类别统计
        # ----------------------
        class_count[label] += 1
        total_boxes += 1

        current_image_classes.add(label)

    # 图片出现统计
    for cls in current_image_classes:
        image_count[cls] += 1

# ==========================
# 输出结果
# ==========================
print("\n" + "=" * 50)
print("数据集标签统计")
print("=" * 50)

for cls in sorted(class_count.keys()):

    print(f"\n类别: {cls}")
    print(f"标签数: {class_count[cls]}")
    print(f"出现图片数: {image_count[cls]}")
    print(f"小目标数(<{SMALL_AREA}px²): "
          f"{small_object_count[cls]}")

print("\n" + "=" * 50)
print(f"总标签数: {total_boxes}")
print(f"空标图片: {empty_images}")
print(f"异常JSON: {bad_json}")
print(f"类别总数: {len(class_count)}")

# 检查异常类别
expected_classes = {
    "battery",
    "board",
    "fire"
}

found_classes = set(class_count.keys())

unknown = found_classes - expected_classes

if len(unknown) > 0:
    print("\n⚠️ 发现异常类别:")
    for x in unknown:
        print(x)

print("\n统计完成")
