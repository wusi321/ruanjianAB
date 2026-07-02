import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# └── 路径配置 └───────────────────────────
DATA_DIR = r"D:\ruanjiain\bbbbbb\sample100"
OUT_DIR  = r"D:\ruanjiain\bbbbbb\output"

os.makedirs(OUT_DIR, exist_ok=True)

# 按 basename 配对：sample_001.xml → sample_001.jpg
xml_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".xml")])

for xml_file in tqdm(xml_files, desc="Processing"):

    xml_path = os.path.join(DATA_DIR, xml_file)

    base, _ = os.path.splitext(xml_file)
    img_name = base + ".jpg"
    img_path = os.path.join(DATA_DIR, img_name)

    try:
        root = ET.parse(xml_path).getroot()

        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图片（跳过）: {img_path}")
            continue

        for obj in root.findall("object"):

            cls = obj.find("name").text.strip()

            bbox = obj.find("bndbox")

            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            area = (xmax - xmin) * (ymax - ymin)

            if cls == "firebig":
                color = (0, 0, 255)
                thickness = 3
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

            label = f"{cls} ({area})"

            text_y = ymin - 8
            if text_y < 20:
                text_y = ymin + 20

            cv2.putText(img, label, (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        save_path = os.path.join(OUT_DIR, img_name)
        cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"处理失败 {xml_file}: {e}")

print(f"\n完成，共输出 {len(xml_files)} 张图片")
print(f"输出目录：{OUT_DIR}")
