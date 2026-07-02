"""
推理后处理：从 YOLO 检测结果中取最大框，输出 firebig
"""

import os
import sys
import cv2
import glob
from ultralytics import YOLO


def infer_firebig(
    image_dir,
    output_dir,
    model_path=r"D:\ruanjiain\bbbbbb\yolo_model\fire\weights\best.pt",
    conf=0.25,
):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    images = sorted(images)

    for img_path in images:
        results = model(img_path, conf=conf, verbose=False)[0]

        # 找到面积最大的框
        best_box = None
        best_area = 0
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

        # 绘制结果
        img = cv2.imread(img_path)
        if best_box:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"firebig ({best_area})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, img)

    print(f"完成！输出至: {output_dir}")


def infer_firebig_xml(
    image_dir,
    output_xml_dir,
    model_path=r"D:\ruanjiain\bbbbbb\yolo_model\fire\weights\best.pt",
    conf=0.25,
):
    """生成竞赛格式的 Pascal VOC XML（仅 firebig）"""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    os.makedirs(output_xml_dir, exist_ok=True)
    model = YOLO(model_path)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    images = sorted(images)

    for img_path in images:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        results = model(img_path, conf=conf, verbose=False)[0]

        # 最大框
        best_box = None
        best_area = 0
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

        # 写 XML
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "captures"
        ET.SubElement(annotation, "filename").text = os.path.basename(img_path)

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"

        if best_box:
            x1, y1, x2, y2 = best_box
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = "firebig"
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bb = ET.SubElement(obj, "bndbox")
            ET.SubElement(bb, "xmin").text = str(x1)
            ET.SubElement(bb, "ymin").text = str(y1)
            ET.SubElement(bb, "xmax").text = str(x2)
            ET.SubElement(bb, "ymax").text = str(y2)

        rough = ET.tostring(annotation, encoding="utf-8")
        dom = minidom.parseString(rough)
        pretty = dom.toprettyxml(indent="  ", encoding="utf-8")

        name = os.path.splitext(os.path.basename(img_path))[0] + ".xml"
        with open(os.path.join(output_xml_dir, name), "wb") as f:
            f.write(pretty)

    print(f"完成！XML 输出至: {output_xml_dir}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        infer_firebig_xml(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output_xml")
    else:
        print("用法: python infer_firebig.py <图片文件夹> [输出XML文件夹]")
#*** End of File
