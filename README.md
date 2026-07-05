# PPYOLOE 目标检测竞赛训练记录

## 项目简介

本项目基于 PaddleDetection 框架，使用 PPYOLOE 进行目标检测训练，适用于：

* AIStudio / Linux 环境
* PaddlePaddle GPU 训练
* COCO 格式数据集
* 小样本目标检测竞赛

当前数据集规模：

* train：324 张有效图像
* val：52 张有效图像
* 类别数：3 类

---

# 1. 环境信息

## 系统环境

| 项目           | 信息       |
| ------------ | -------- |
| 平台           | AIStudio |
| Python       | 3.10     |
| PaddlePaddle | 3.3.0    |
| CUDA         | 可用       |
| GPU 数量       | 1        |
| Runtime CUDA | 11.8     |
| Driver API   | 12.0     |

验证代码：

```python
import paddle

print("Paddle:", paddle.__version__)
print("CUDA available:", paddle.is_compiled_with_cuda())
print("GPU count:", paddle.device.cuda.device_count())
```

输出：

```python
Paddle: 3.3.0
CUDA available: True
GPU count: 1
```

---

# 2. 项目目录结构

当前目录：

```bash
/home/aistudio
```

项目结构：

```text
/home/aistudio
│
├── PaddleDetection/
│
├── competition/
│   └── dataset/
│       ├── train/
│       │   └── images/
│       │
│       ├── val/
│       │   └── images/
│       │
│       └── annotations/
│           ├── train.json
│           └── val.json
│
├── output/
│
└── configs/
    └── competition/
        └── my_ppyoloe.yml
```

---

# 3. 使用模型

模型：

```text
PPYOLOE+
```

基础配置：

```yaml
_BASE_: [
  '../ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml'
]
```

预训练模型：

```yaml
weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_plus_crn_s_obj365_pretrained.pdparams
```

---

# 4. 数据集格式

采用 COCO 格式。

标注文件：

```text
annotations/train.json
annotations/val.json
```

类别数：

```yaml
num_classes: 3
```

---

# 5. 最终稳定版 YAML 配置

文件路径：

```text
/home/aistudio/PaddleDetection/configs/competition/my_ppyoloe.yml
```

配置如下：

```yaml
_BASE_: [
  '../ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml'
]

find_unused_parameters: False

log_iter: 20
snapshot_epoch: 5

metric: COCO
num_classes: 3

epoch: 40
worker_num: 0
use_gpu: true

save_dir: /home/aistudio/output

weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_plus_crn_s_obj365_pretrained.pdparams


PPYOLOEHead:
  num_classes: 3


TrainDataset:
  !COCODataSet
    image_dir: train/images
    anno_path: annotations/train.json
    dataset_dir: /home/aistudio/competition/dataset
    data_fields: ['image', 'gt_bbox', 'gt_class']
    allow_empty: true
    empty_ratio: 0.2


EvalDataset:
  !COCODataSet
    image_dir: val/images
    anno_path: annotations/val.json
    dataset_dir: /home/aistudio/competition/dataset


TestDataset:
  !ImageFolder
    anno_path: none
    dataset_dir: /home/aistudio/competition/dataset


LearningRate:
  base_lr: 0.0005

  schedulers:
    - !CosineDecay
      max_epochs: 40

    - !LinearWarmup
      start_factor: 0.0
      epochs: 3


OptimizerBuilder:
  clip_grad_by_norm: 35.


TrainReader:
  batch_size: 4
  shuffle: true
  drop_last: true

  sample_transforms:
    - Decode: {}

    - RandomFlip:
        prob: 0.5

    - RandomDistort: {}

    - Resize:
        target_size: [640, 640]
        keep_ratio: false
        interp: 2

    - NormalizeImage:
        is_scale: true
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]

    - Permute: {}

  batch_transforms:
    - PadGT:
        return_gt_mask: true


EvalReader:
  batch_size: 2

  sample_transforms:
    - Decode: {}

    - Resize:
        target_size: [640, 640]
        keep_ratio: false
        interp: 2

    - NormalizeImage:
        is_scale: true
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]

    - Permute: {}


TestReader:
  batch_size: 1

  sample_transforms:
    - Decode: {}

    - Resize:
        target_size: [640, 640]
        keep_ratio: false
        interp: 2

    - NormalizeImage:
        is_scale: true
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]

    - Permute: {}
```

---

# 6. 开始训练

训练命令：

```bash
python3 /home/aistudio/PaddleDetection/tools/train.py \
-c /home/aistudio/PaddleDetection/configs/competition/my_ppyoloe.yml \
--eval
```

---

# 7. 恢复训练

恢复训练：

```bash
python3 /home/aistudio/PaddleDetection/tools/train.py \
-c /home/aistudio/PaddleDetection/configs/competition/my_ppyoloe.yml \
-r /home/aistudio/output/9 \
--eval
```

注意：

```bash
-r /home/aistudio/output/9
```

不要写：

```bash
-r /home/aistudio/output/9.pdparams
```

因为 Paddle 会自动读取：

```text
9.pdparams
9.pdopt
9.pdema
```

---

# 8. 当前训练结果

## Epoch 4

```text
AP@[IoU=0.50:0.95] = 0.609
```

## Epoch 9

```text
AP@[IoU=0.50:0.95] = 0.648
```

说明模型已经正常收敛。

---

# 9. 已踩坑记录

---

## 9.1 train.py 参数错误

错误：

```text
train.py: error: unrecognized arguments
```

原因：

Notebook 中换行符写法错误：

错误：

```bash
\ -c
```

正确：

```bash
\
-c
```

---

## 9.2 数据集路径错误

错误：

```text
Dataset is not valid
```

原因：

```yaml
anno_path
dataset_dir
```

路径配置错误。

---

## 9.3 OpenCV resize 崩溃

错误：

```text
cv2.error: resize.cpp
!dsize.empty()
```

原因：

随机 resize + 数据异常。

解决：

固定：

```yaml
Resize:
  target_size: [640,640]
```

关闭：

```yaml
random_size: False
```

---

## 9.4 PAN concat shape mismatch

错误：

```text
136 != 135
```

原因：

多尺度训练导致 feature map 奇数尺寸。

解决：

固定输入：

```yaml
target_size: [640,640]
keep_ratio: false
```

---

## 9.5 DataLoader 崩溃

错误：

```text
Tensor holds no memory
Blocking queue is killed
```

原因：

Paddle 多 worker 不稳定。

解决：

```yaml
worker_num: 0
```

---

## 9.6 YAML 配置损坏

错误：

```text
TypeError: argument of type 'NoneType'
```

原因：

yaml 缩进/格式损坏。

建议：

使用 Python 自动写入 yaml 文件。

---

# 10. checkpoint 文件说明

output 目录：

```text
output/
├── 4.pdparams
├── 4.pdopt
├── 4.pdema
├── 9.pdparams
├── 9.pdopt
├── 9.pdema
├── best_model.pdparams
├── best_model.pdopt
├── best_model.pdema
```

说明：

| 文件       | 作用     |
| -------- | ------ |
| pdparams | 模型权重   |
| pdopt    | 优化器状态  |
| pdema    | EMA 权重 |

---

# 11. 当前训练建议

当前数据量：

```text
324 train
52 val
```

属于小数据集。

建议：

| 参数         | 推荐  |
| ---------- | --- |
| epoch      | 40  |
| batch_size | 4   |
| 输入尺寸       | 640 |
| worker_num | 0   |

预计：

| Epoch | AP    |
| ----- | ----- |
| 10    | 0.64+ |
| 15    | 0.68+ |
| 20    | 0.70+ |

---

# 12. 后续优化方向

## 可继续尝试

### 1. TTA 测试增强

例如：

* flip
* multi-scale

---

### 2. 更大模型

可尝试：

```text
PPYOLOE-L
RT-DETR
```

---

### 3. 数据增强

可增加：

* Mosaic
* MixUp
* CopyPaste

但小数据集需谨慎。

---

### 4. 伪标签

可使用：

```text
test → pseudo label → retrain
```

提升 leaderboard。

---

# 13. 常用命令

查看 GPU：

```bash
nvidia-smi
```

查看 output：

```bash
ls /home/aistudio/output
```

查看 yaml：

```bash
cat /home/aistudio/PaddleDetection/configs/competition/my_ppyoloe.yml
```

继续训练：

```bash
python3 tools/train.py -c xxx.yml -r output/9 --eval
```

---

# 14. 提交推理脚本 (py/predict.py)

## 文件位置

```text
d:\桌面\output\py\predict.py
```

## 设计说明

此脚本基于官方模板，修复了旧脚本导致 F1 Score = 0 的 3 个关键问题：

| # | 问题 | 旧值 | 新值 | 原因 |
|---|------|------|------|------|
| 1 | `switch_ir_optim` | True | False | IR优化改变输出张量名称/顺序，导致选错输出 |
| 2 | `switch_use_feed_fetch_ops` | 未设置 | False | Paddle 3.x 兼容 |
| 3 | 输出张量解析 | 按 shape [N,6] 模糊匹配 | 按 output_names 名称索引 | 官方模板标准方式 |

## 模型导出（必须先做）

```bash
python /home/aistudio/PaddleDetection/tools/export_model.py \
    -c /home/aistudio/PaddleDetection/configs/competition/my_ppyoloe.yml \
    -o weights=/home/aistudio/output/best_model.pdparams \
    --output_dir=/home/aistudio/output_inference
```

导出后的 `output_inference/` 目录自动包含：
- `model.pdmodel`
- `model.pdiparams`
- `infer_cfg.yml`（自动生成，arch=PPYOLOE, label_list=[battery, board, fire]）

## 提交包结构

```text
submission.zip
├── predict.py              ← 使用 py/predict.py
├── model/
│   ├── infer_cfg.yml       ← 导出自动生成的正确版本
│   ├── model.pdmodel
│   └── model.pdiparams
└── PaddleDetection/
    └── deploy/python/
        ├── preprocess.py
        └── utils.py
```

---

# 15. 总结

当前训练环境已经稳定：

* 数据集正常
* GPU 正常
* checkpoint 正常
* 恢复训练正常
* AP 已达到 0.648

当前主要目标：

1. 稳定训练到 15~20 epoch
2. 观察 best AP
3. 避免过拟合
4. 尝试 TTA 与推理优化

整体训练流程已可用于正式竞赛。
