import json
import os
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom


def labelme_json_to_voc_xml(json_path, output_dir):
    """Convert LabelMe JSON to Pascal VOC XML format"""
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_name = data.get("imagePath", "unknown.jpg")
    img_width = data.get("imageWidth", 0)
    img_height = data.get("imageHeight", 0)
    img_depth = data.get("imageDepth", 3)

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "firedetect_public"
    ET.SubElement(annotation, "filename").text = os.path.basename(img_name)

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = str(img_depth)

    for shape in data.get("shapes", []):
        label = shape.get("label", "firebig").strip()
        points = shape.get("points", [])

        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    rough = ET.tostring(annotation, encoding="utf-8")
    dom = minidom.parseString(rough)
    pretty = dom.toprettyxml(indent="  ", encoding="utf-8")

    xml_name = os.path.splitext(os.path.basename(json_path))[0] + ".xml"
    out_path = os.path.join(output_dir, xml_name)
    with open(out_path, "wb") as f:
        f.write(pretty)

    print(f"Converted: {json_path} -> {out_path}")


def batch_convert(json_dir, output_dir):
    """Batch convert all LabelMe JSON files in directory"""
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in: {json_dir}")
        return
    for jf in json_files:
        labelme_json_to_voc_xml(jf, output_dir)
    print(f"\nConverted {len(json_files)} files, output to: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        batch_convert(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        labelme_json_to_voc_xml(sys.argv[1], ".")
    else:
        print("Usage:")
        print("  Batch: python json2voc.py <LabelMe JSON dir> <output dir>")
        print("  Single: python json2voc.py <file.json>")
