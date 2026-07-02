import os
import cv2
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ── 配置 ──────────────────────────────────────────
SAMPLE_DIR  = r"D:\ruanjiain\bbbbbb\sample100"   # 100张样标
YOLO_DIR    = r"D:\ruanjiain\bbbbbb\yolo_data"    # YOLO 训练数据目录
MODEL_DIR   = r"D:\ruanjiain\bbbbbb\yolo_model"   # 模型保存
EPOCHS      = 100
IMG_SIZE    = 640

# ── 1. 转换 XML → YOLO TXT ──────────────────────
def convert_xml_to_yolo(xml_path, out_txt_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in classes:
            classes[name] = len(classes)

        xmin = int(float(obj.find("bndbox/xmin").text))
        ymin = int(float(obj.find("bndbox/ymin").text))
        xmax = int(float(obj.find("bndbox/xmax").text))
        ymax = int(float(obj.find("bndbox/ymax").text))

        # YOLO 格式: class_id x_center y_center width height (归一化)
        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h
        lines.append(f"{classes[name]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines))


# ── 2. 准备 YOLO 数据集 ──────────────────────────
def prepare_dataset():
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)

    classes = {}
    xml_files = sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith(".xml")])

    images = []
    for xml_file in xml_files:
        base = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(SAMPLE_DIR, xml_file)
        jpg_path = os.path.join(SAMPLE_DIR, base + ".jpg")
        if not os.path.exists(jpg_path):
            continue

        # 按 8:2 分训练/验证
        images.append((jpg_path, xml_path, base))

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for split_name, split_list in [("train", train_imgs), ("val", val_imgs)]:
        img_dir = os.path.join(YOLO_DIR, "images", split_name)
        lbl_dir = os.path.join(YOLO_DIR, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for jpg_path, xml_path, base in split_list:
            # 复制图片
            shutil.copy(jpg_path, os.path.join(img_dir, base + ".jpg"))
            # 转换标注
            convert_xml_to_yolo(xml_path, os.path.join(lbl_dir, base + ".txt"), classes)

    # 写 dataset.yaml
    yaml_path = os.path.join(YOLO_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {list(classes.keys())}\n")

    print(f"数据集准备完成：{len(train_imgs)} 训练 / {len(val_imgs)} 验证")
    print(f"类别: {classes}")
    return yaml_path


# ── 3. 训练 YOLO 模型 ────────────────────────────
def train_model(yaml_path):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")  # 用 Nano 版，CPU 也能跑
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=MODEL_DIR,
        name="fire",
        device="cpu",
        patience=30,       # 30轮没提升就早停
        verbose=True,
    )
    print(f"训练完成，模型保存至: {MODEL_DIR}/fire/weights/best.pt")
    return f"{MODEL_DIR}/fire/weights/best.pt"


if __name__ == "__main__":
    print("=" * 50)
    print("步骤 1/2：准备 YOLO 数据集")
    print("=" * 50)
    yaml_path = prepare_dataset()

    print("\n" + "=" * 50)
    print("步骤 2/2：训练模型（约 10-20 分钟，视 CPU 性能）")
    print("=" * 50)
    model_path = train_model(yaml_path)
    print(f"\n模型已保存: {model_path}")
    print("\n下一步：用 python prelabel_yolo.py 对剩余图片做预标注")
*** End of File
