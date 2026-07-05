# MS 项目工程大纲

## 项目概述
基于 PaddlePaddle 的目标检测推理项目，使用双模型（M/S）融合策略进行电池(battery)、主板(board)、火焰(fire)三类目标的检测。

---

## 文件结构

```
MS/
├── MS.log                          # 项目变更日志
├── MS.md                           # 项目工程大纲（本文件）
├── submission/
│   └── submission/
│       ├── predict.py              # 【核心】主推理脚本，包含模型加载、预处理、推理、NMS、双模型融合
│       ├── model/                   # S 模型目录（小模型）
│       │   ├── model.pdmodel        #   Paddle 模型结构文件
│       │   ├── model.pdiparams      #   Paddle 模型参数文件
│       │   └── infer_cfg.yml        #   推理配置文件
│       ├── model_m/                 # M 模型目录（大模型）
│       │   ├── model.pdmodel        #   Paddle 模型结构文件
│       │   ├── model.pdiparams      #   Paddle 模型参数文件
│       │   └── infer_cfg.yml        #   推理配置文件
│       └── PaddleDetection/         # PaddleDetection 工具代码
│           └── deploy/python/
│               ├── preprocess.py    #   预处理工具
│               ├── utils.py         #   通用工具函数
│               └── keypoint_preprocess.py  # 关键点预处理
└── submission删打印/                # 备份版本（去除打印语句的版本）
    └── ...
```

---

## 核心文件说明

### predict.py — 主推理脚本
| 功能模块 | 行号范围 | 说明 |
|---------|---------|------|
| 路径定义 | L19-L28 | `MODEL_DIR_S`(S模型→model/), `MODEL_DIR_M`(M模型→model_m/) |
| 参数配置 | L32-L41 | `INPUT_SIZE=640`, `SCORE_THRESH=0.4`, CLASS_MAP 类别映射 |
| IOU 计算 | L43-L58 | 两个边界框的交并比计算（保留备用） |
| NMS 非极大抑制 | L61-L110 | 按类别进行非极大值抑制，默认阈值0.37 |
| build_index | L112-L117 | 按类别建立检测结果索引（setdefault） |
| fusion_results | L119-L136 | M/S 结果融合：M为基础，S高置信度框补充。用abs距离(<50px)替代IOU，末尾NMS阈值0.5 |
| load_predictor | L170-L206 | 加载 PaddlePaddle GPU 推理模型 |
| preprocess | L210-L257 | 图像预处理：resize→归一化→标准化(Normalize) |
| _predict_array | L290-L488 | 核心推理函数：接收numpy数组+image_id，支持ROI推理 |
| predict_image | L491-L504 | 文件包装函数：读取图片→调用_predict_array |
| main | L505-L608 | 固定流水线：M全图→筛低置信框(score<0.6)→S只跑ROI(±30px裁剪)→fusion |

### 模型目录
- **model/** — S 模型（小模型），路径变量 `MODEL_DIR_S`
- **model_m/** — M 模型（大模型），路径变量 `MODEL_DIR_M`

### 融合策略 (fusion_results)
- M 模型全图推理为基础（全量保留）
- S 模型只对 M 低置信框(score<0.6)的 ROI 区域(±30px裁剪)推理
- S 结果中 score >= 0.69 且与 M 框 x/y 差值 < 50px 的视为已覆盖，跳过
- 未覆盖的 S 高置信框作为补充加入
- 最终 NMS 阈值 0.5（轻量）

### 推理流水线 (main)
```
M模型: 全图检测 (predictor_m)
    ↓
筛低置信框 (score < 0.6)
    ↓
S模型: 只跑ROI区域 (predictor_s, ±30px扩边裁剪)
    ↓
fusion_results (M基础, S补充, abs距离匹配)
    ↓
NMS 0.5
```
