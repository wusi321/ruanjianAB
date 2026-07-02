
import os
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# 配置
# =========================

ROOT = r"D:\ruanjiain\bbbbbb\firedetect_public"

IMG_DIR = os.path.join(ROOT, "images")
XML_DIR = os.path.join(ROOT, "annotations")

OUT_ROOT = r"D:\ruanjiain\bbbbbb\firedetect_public\coco_dataset"

TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# =========================

random.seed(RANDOM_SEED)

train_img_dir = os.path.join(OUT_ROOT, "train/images")
val_img_dir = os.path.join(OUT_ROOT, "val/images")

anno_dir = os.path.join(OUT_ROOT, "annotations")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(anno_dir, exist_ok=True)

xml_files = sorted(
    [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
)

random.shuffle(xml_files)

split_idx = int(len(xml_files) * TRAIN_RATIO)

train_files = xml_files[:split_idx]
val_files = xml_files[split_idx:]


def convert(xml_list, image_out_dir, json_path):

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "fire",
                "supercategory": "fire"
            }
        ]
    }

    image_id = 1
    ann_id = 1

    for xml_name in xml_list:

        xml_path = os.path.join(XML_DIR, xml_name)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text

        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        src_img = os.path.join(IMG_DIR, filename)
        dst_img = os.path.join(image_out_dir, filename)

        if not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

        for obj in root.findall("object"):

            cls_name = obj.find("name").text

            if cls_name not in ["fire", "firebig"]:
                continue

            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            w = xmax - xmin
            h = ymax - ymin

            area = w * h

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
            )

            ann_id += 1

        image_id += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)


convert(
    train_files,
    train_img_dir,
    os.path.join(anno_dir, "train.json")
)

convert(
    val_files,
    val_img_dir,
    os.path.join(anno_dir, "val.json")
)

print("=================================")
print("Train Images:", len(train_files))
print("Val Images:", len(val_files))
print("Train Json :", os.path.join(anno_dir, "train.json"))
print("Val Json   :", os.path.join(anno_dir, "val.json"))
print("Done.")
