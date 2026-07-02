import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ==========================
# 路径配置
# ==========================
IMG_DIR  = r"D:\ruanjiain\bbbbbb\firedetect_public\images"
XML_DIR  = r"D:\ruanjiain\bbbbbb\firedetect_public\annotations"
OUT_DIR  = r"D:\ruanjiain\bbbbbb\firedetect_public\output"

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# 处理
# ==========================
xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith(".xml")])

for xml_file in tqdm(xml_files, desc="Drawing boxes"):

    base, _ = os.path.splitext(xml_file)
    img_path = os.path.join(IMG_DIR, base + ".jpg")
    xml_path = os.path.join(XML_DIR, xml_file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"跳过（无图片）: {img_path}")
        continue

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        print(f"解析失败 {xml_file}: {e}")
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
            color = (0, 0, 255)       # 红色，粗线
            thickness = 3
        else:
            color = (0, 255, 0)       # 绿色，细线
            thickness = 2

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        label = "{} ({})".format(cls, area)
        text_y = ymin - 8
        if text_y < 20:
            text_y = ymin + 20
        cv2.putText(img, label, (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(OUT_DIR, base + ".jpg"), img)

print("\n完成！共输出 {} 张图片".format(len(xml_files)))
print("输出目录: {}".format(OUT_DIR))
