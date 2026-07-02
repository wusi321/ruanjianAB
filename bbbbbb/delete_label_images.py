#```python
# -*- coding: utf-8 -*-

import os

# label 文件夹路径
LABEL_DIR = r"D:\ruanjiain\A_train\label"

# 要删除的图片后缀
IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

deleted = 0

for filename in os.listdir(LABEL_DIR):

    path = os.path.join(
        LABEL_DIR,
        filename
    )

    # 跳过文件夹
    if not os.path.isfile(path):
        continue

    ext = os.path.splitext(filename)[1].lower()

    # 删除图片
    if ext in IMG_EXTS:
        try:
            os.remove(path)
            deleted += 1
            print(f"删除: {filename}")

        except Exception as e:
            print(f"失败: {filename}")
            print(e)

print("\n完成")
print(f"共删除 {deleted} 张图片")
print("json 标注文件已保留")
