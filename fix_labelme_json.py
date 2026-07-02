#```python
# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm

LABEL_DIR = r"D:\ruanjiain\firedetect_public\annotations"

json_files = [
    f for f in os.listdir(LABEL_DIR)
    if f.endswith(".json")
]

fixed_count = 0

for jf in tqdm(json_files):

    path = os.path.join(LABEL_DIR, jf)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False

    for shape in data.get("shapes", []):

        pts = shape.get("points", [])

        # 只修 rectangle + 4点
        if (
            shape.get("shape_type") == "rectangle"
            and len(pts) == 4
        ):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)

            # 改成新版labelme格式
            shape["points"] = [
                [x1, y1],
                [x2, y2]
            ]

            changed = True

    # 修 imagePath
    jpg_name = os.path.splitext(jf)[0] + ".jpg"
    data["imagePath"] = jpg_name

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        fixed_count += 1

print(f"修复完成，共处理 {fixed_count} 个 json")
