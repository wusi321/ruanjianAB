import os

# =========================================================
# 修改这里
# =========================================================

IMAGE_DIR = "/home/aistudio/competition/dataset/val/images"

SAVE_TXT = "/home/aistudio/data.txt"

# =========================================================
# 扫描图片
# =========================================================

image_list = []

for file_name in sorted(os.listdir(IMAGE_DIR)):

    if file_name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):

        full_path = os.path.join(
            IMAGE_DIR,
            file_name
        )

        image_list.append(full_path)

# =========================================================
# 写入 txt
# =========================================================

with open(SAVE_TXT, "w") as f:

    for path in image_list:

        f.write(path + "\n")

print("=" * 50)
print(f"Total Images: {len(image_list)}")
print(f"Saved: {SAVE_TXT}")
print("=" * 50)