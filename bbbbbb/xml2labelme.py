import os, json, cv2, xml.etree.ElementTree as ET
from glob import glob

XML_DIR = r"D:\ruanjiain\bbbbbb\firedetect_public\annotations"
IMG_DIR = r"D:\ruanjiain\bbbbbb\firedetect_public\images"
OUT_DIR = r"D:\ruanjiain\bbbbbb\firedetect_public\LabelMe"
os.makedirs(OUT_DIR, exist_ok=True)

xml_files = sorted(glob(os.path.join(XML_DIR, "*.xml")))

for xml_path in xml_files:
    base = os.path.splitext(os.path.basename(xml_path))[0]
    img_path = os.path.join(IMG_DIR, base + ".jpg")

    # 读取图片获取尺寸
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
    else:
        h, w = 480, 640  # fallback

    root = ET.parse(xml_path).getroot()

    shapes = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        x1 = int(float(bbox.find("xmin").text))
        y1 = int(float(bbox.find("ymin").text))
        x2 = int(float(bbox.find("xmax").text))
        y2 = int(float(bbox.find("ymax").text))

        shapes.append({
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        })

    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": base + ".jpg",
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "imageDepth": 3
    }

    with open(os.path.join(OUT_DIR, base + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("完成！共转换 {} 个文件到 {}".format(len(xml_files), OUT_DIR))
