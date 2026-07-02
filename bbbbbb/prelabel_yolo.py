"""
用训练好的 YOLO 模型对大量图片做预标注，
输出 LabelMe JSON 格式（可在 LabelMe 中打开微调）。
"""

import os
import sys
import json
import glob
import cv2
from ultralytics import YOLO


def prelabel_images(
    image_dir,
    output_dir,
    model_path=r"D:\ruanjiain\bbbbbb\yolo_model\fire\weights\best.pt",
    conf=0.25,
):
    """用 YOLO 模型预标注图片，输出 LabelMe JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model = YOLO(model_path)

    # 找所有图片
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    images = sorted(images)

    print(f"共 {len(images)} 张图片，开始预标注...\n")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        h, w = img.shape[:2]

        # YOLO 推理
        results = model(img_path, conf=conf, verbose=False)[0]

        shapes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            shapes.append({
                "label": cls_name,
                "points": [[x1, y1], [x2, y2]],
            })

        # 写 JSON
        data = {
            "version": "5.0.1",
            "flags": {},
            "imagePath": os.path.basename(img_path),
            "imageHeight": h,
            "imageWidth": w,
            "imageDepth": 3,
            "shapes": shapes,
        }

        name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
        out_path = os.path.join(output_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        status = f"检出 {len(shapes)} 个" if shapes else "无结果"
        print(f"[{status}] {os.path.basename(img_path)}")

    print(f"\n完成！输出至: {output_dir}")
    print("用 LabelMe 打开此目录，JSON 会自动加载为预标注框。")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        img_dir = sys.argv[1]
        out_dir = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else r"D:\ruanjiain\bbbbbb\yolo_model\fire\weights\best.pt"
        prelabel_images(img_dir, out_dir, model_path)
    else:
        print("用法: python prelabel_yolo.py <图片文件夹> <输出JSON文件夹> [模型路径]")
        print("示例: python prelabel_yolo.py D:\\images D:\\prelabels")
*** End of File
