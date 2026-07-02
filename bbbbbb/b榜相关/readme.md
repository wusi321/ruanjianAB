# 2026软件杯文心小汪赛道赛题
本项目基于 PaddleDetection2.5 框架，使用 PPYOLOE 进行目标检测训练，适用于：

* AIStudio / Linux 环境
* PaddlePaddle GPU 训练
* COCO 格式数据集
* 小样本目标检测竞赛

当前数据集规模：

项目	内容
名称	火焰检测公开数据集（firedetect_public）
任务	目标检测（火焰定位）
图像数	3030 张
分辨率	640 × 480，RGB，JPEG
类别数	2（fire、firebig）
标注格式	Pascal VOC（XML，逐图多目标）
标注框总数	8740（firebig 3030 + fire 5710）
来源	利用端侧AR渲染引擎合成的增强现实室内火灾场景
标注方式	渲染期自动生成（像素级，非人工框选）


类别	含义	每图数量
fire	画面中的普通火焰（除最大者以外的其它可见火）	0 个或多个
firebig	画面中最主要（面积最大）的一处火焰	至多 1 个
每张图都至少含 1 个 firebig（无可见火的帧已剔除，不在数据集中）。
若画面只有一处火，则它即为 firebig，此时无 fire 框。
"最大"按火焰在画面中的可见像素面积判定（近大远小）。

项目结构：

```text
/home/aistudio
│
├── PaddleDetection/   #2.5.0
│           └── ompetition/configs/
│               └── competition/
│                    └── my_ppyoloe.yml
│
├── competition         #仅标注了部分
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

```
实际情况说明
采用模型全部输出fire，后处理计算最大框为firebig的方案
官方给的数据集为去除干扰的，本地标注后也无干扰，训练结果可能本地测试高
现在的训练集包含500+无氛围图片，和300+带干扰的图片，手动标注
标注工具为 labelme 输出json格式，脚本转换为xml格式，上传后需要转换为coco格式

现在格式
firedetect_public/
├── annotations/
│     ├──  ....xml
│
├── images/
│     ├──  ....jpg

上传后转化coco格式

├── competition         #仅标注了部分
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

