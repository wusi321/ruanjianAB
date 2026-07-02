# 软件杯 — 火灾检测（PPYOLOE）

本仓库为 **软件杯** 赛题的本地代码与数据集仓库，基于 **PaddleDetection PPYOLOE** 模型，围绕火灾检测任务进行数据处理、模型训练、推理与结果分析。

---

## 目录结构

| 目录/文件 | 说明 |
|-----------|------|
| `A_train/` | A 榜训练集（图像 + JSON 标注） |
| `A_train_out/` | A 榜官方输出/标注结果 |
| `A_trainbu/` | A 榜补充数据 |
| `analysis_result/` | 标注风格分析输出 |
| `firedetect_public/` | 公开火灾检测数据集 |
| `bbbbbb/` | 核心工作目录，包含 PaddleDetection、模型权重、多版本数据集及脚本 |
| `bbbbbb/PaddleDetection/` | PPYOLOE 模型代码与预训练权重 |
| `bbbbbb/oott/` | 多次提交的推理结果与模型备份 |

---

## 脚本说明

### 数据处理

| 脚本 | 用途 |
|------|------|
| `bbbbbb/json2voc.py` | 将 LabelMe JSON 标注转为 Pascal VOC XML 格式 |
| `bbbbbb/voc2coco_fire_all_local.py` | 将 VOC 格式数据集转为 COCO 格式，支持训练/验证集划分 |
| `bbbbbb/xml2labelme.py` | 将 VOC XML 标注转回 LabelMe JSON 格式 |
| `fix_labelme_json.py` | 修复 LabelMe JSON 文件中异常的标注形状 |
| `delete_label_images.py` | 删除 A 榜数据中图片/标注不匹配的冗余图片 |

### 检测与训练

| 脚本 | 用途 |
|------|------|
| `bbbbbb/train_yolo.py` | YOLO 模型训练，支持 XML 转 YOLO 格式、数据集划分 |
| `bbbbbb/prelabel_yolo.py` | 用训练好的 YOLO 模型对新图片预标注，输出 LabelMe JSON |
| `bbbbbb/infer_firebig.py` | 推理后处理，从 YOLO 检测结果中提取最大火点框 |

### 分析与可视化

| 脚本 | 用途 |
|------|------|
| `analyze_annotation_style.py` | 对比本地标注与官方标注的风格差异 |
| `analyze_style.py` | 分析标注风格的基础统计 |
| `count_labels.py` | 统计各类别目标数量、小目标比例等 |
| `imge_xianshi.py` | 在图片上绘制标注框，快速检查标注质量 |
| `vis_labelme.py` | 可视化 LabelMe JSON 标注框 |
| `bbbbbb/vis_xml2img.py` | 将 XML 标注绘制到对应图片上 |

---

## 环境要求

- Python 3.8+
- PaddleDetection / PaddlePaddle
- YOLOv5 / Ultralytics
- OpenCV、NumPy、Pandas、Matplotlib、tqdm

---

## 快速开始

1. 安装 PaddlePaddle 与 PaddleDetection
2. 使用 `voc2coco_fire_all_local.py` 将 VOC 标注转为 COCO 格式
3. 使用 PaddleDetection 下的 PPYOLOE 配置文件训练
4. 使用 `train_yolo.py` 或 PPYOLOE 原生流程完成模型训练
5. 使用 `prelabel_yolo.py` / `infer_firebig.py` 对新数据推理
