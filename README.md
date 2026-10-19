# VisDrone 无人机航拍图目标检测系统

基于 Ultralytics YOLO26n 与 VisDrone2019-DET 数据集的无人机航拍场景目标检测系统，支持数据集预处理、模型训练、单张 / 批量推理以及 Streamlit 可视化 Web 界面。

## 1. 主要功能

- **数据集预处理**
  - `data_split.py`：将 VisDrone 原始数据集按 8:1:1 随机划分为训练集 / 验证集 / 测试集（随机种子固定为 42，划分结果可复现）。
  - `format_transforms.py`：将 VisDrone 原始标注格式（`x, y, w, h, score, category, ...`）就地转换为 YOLO 归一化格式（`class x_center y_center w h`），自动裁剪越界框并过滤无效类别。
- **模型训练**
  - `train.py`：基于预训练权重 `yolo26n.pt` 在 VisDrone 数据集上微调训练，训练配置由 `Visdrone.yaml` 管理。
- **模型推理**
  - `test_one.py`：对单张图片进行推理，弹出窗口显示并保存画框结果。
  - `test_more.py`：对 `test_images/` 目录下的图片进行批量推理。
- **可视化 Web 界面**
  - `streamlit_app.py`：Streamlit 交互界面，支持上传图片或选择验证集示例图片，可调置信度阈值，实时展示检测框、类别、置信度及目标数量。

### 检测类别（10 类）

| 类别 ID | 名称 | 类别 ID | 名称 |
| :---: | --- | :---: | --- |
| 0 | pedestrian（行人） | 5 | truck（卡车） |
| 1 | people（人群） | 6 | tricycle（三轮车） |
| 2 | bicycle（自行车） | 7 | awning-tricycle（带篷三轮车） |
| 3 | car（轿车） | 8 | bus（公交车） |
| 4 | van（面包车） | 9 | motor（摩托车） |

## 2. 安装方法

### 环境要求

- Python 3.8+
- 可选：NVIDIA GPU + CUDA（训练时 `device=0` 使用 GPU，无 GPU 可改为 `device="cpu"`）

### 安装依赖

本项目基于 Ultralytics（项目根目录已包含 `ultralytics-main` 源码），安装方式二选一：

```bash
# 方式一：直接安装 ultralytics 发行版（推荐）
pip install ultralytics

# 方式二：从项目内源码安装
pip install -e ultralytics-main
```

再安装 Web 界面依赖：

```bash
pip install streamlit opencv-python pillow
```

### 准备数据集

1. 下载 [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) 数据集，放入 `dataset/` 目录，结构如下：

```
dataset/
└── VisDrone2019-DET-train/
    ├── images/        # 原始图片
    └── annotations/   # 原始标注（.txt）
```

2. 运行数据划分与格式转换脚本（详见下一节），生成 `VisDrone_split/` 数据集目录。

## 3. 使用方法

按以下顺序执行即可完成从数据准备到推理的完整流程：

### 步骤 1：划分数据集

```bash
python data_split.py
```

输出 `VisDrone_split/`，包含 `images/train`、`images/val`、`labels/train`、`labels/val`（训练集约 6471 张，验证集约 548 张）。

### 步骤 2：转换标注格式

```bash
python format_transforms.py
```

将 `VisDrone_split/labels/` 下的标注就地覆盖为 YOLO 格式。

### 步骤 3：训练模型

```bash
python train.py
```

训练参数（可在 `train.py` 中修改）：`epochs=10`、`imgsz=640`、`batch=4`、`device=0`。

> Windows 下必须保持 `workers=0`，否则多进程数据加载会报错。

训练完成后，最佳权重保存在 `runs/detect/train-2/weights/best.pt`。

已经训练好的修改后模型，更改为p2_yolo26n.pt保存在主目录，可以在第五步中直接调用

### 步骤 4：推理

**单张图片：**

```bash
python test_one.py
```

对 `test.jpg` 推理，结果显示并保存到 `runs/detect/predict/`。

**批量图片：**

```bash
python test_more.py
```

对 `test_images/` 目录下所有图片推理，结果保存到 `runs/detect/predict-2/`。

**Web 界面：**

```bash
streamlit run streamlit_app.py
```

浏览器打开后，上传图片或在侧边栏选择示例图片，调节置信度阈值（默认 0.25），点击「开始推理」即可查看检测结果。

## 4. 输入输出示例

### 输入

- **图片**：JPG / JPEG / PNG 格式的无人机航拍图，示例见 `test.jpg` 与 `test_images/` 目录。
- **标注（训练用）**：YOLO 格式 `.txt`，每行一个目标：

```
3 0.512305 0.486328 0.045898 0.062500
```

含义：`类别ID 中心x 中心y 宽 高`（均为 0~1 归一化坐标），上例表示一辆轿车（car）。

### 输出

- **推理结果图**：画有绿色检测框的图片，每个框标注 `类别: 置信度`，例如 `car: 0.87`，保存在 `runs/detect/predict*/` 下。
- **Web 界面检测信息**：逐目标列出类别与置信度，并统计目标总数，示例：

```
🎯 目标 1: car (置信度: 0.87)
🎯 目标 2: pedestrian (置信度: 0.76)
✅ 检测完成！共识别到 2 个目标
```

- **训练产物**（`runs/detect/train-2/`、p2_yolo26n.pt）：训练曲线 `results.png`、混淆矩阵 `confusion_matrix.png`、F1 / PR 曲线及权重文件。

### 参考指标（p2_YOLO26n，100 epochs，imgsz=640）

| 指标 | 数值 |
| --- | --- |
| Precision | 0.488 |
| Recall | 0.375 |
| mAP@50 | 0.369 |
| mAP@50-95 | 0.203 |

> 增加训练轮数、增大输入分辨率或使用更大的模型（如 yolo26s/m）可进一步提升精度。

## 项目结构

```
visdrone/
├── Visdrone.yaml          # 数据集配置文件
├── data_split.py          # 数据集划分脚本
├── format_transforms.py   # 标注格式转换脚本（VisDrone → YOLO）
├── train.py               # 训练脚本
├── test_one.py            # 单张图片推理
├── test_more.py           # 批量图片推理
├── streamlit_app.py       # Streamlit 可视化界面
├── yolo26n.pt             # 预训练权重
├── dataset/               # 原始 VisDrone 数据集
├── VisDrone_split/        # 划分并转换后的数据集
├── test_images/           # 批量测试图片
├── weights/               # 权重目录
├── runs/                  # 训练与推理输出
└── ultralytics-main/      # Ultralytics 源码
```

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

> 注意：项目依赖的 Ultralytics 采用 AGPL-3.0 许可证（见 `ultralytics-main/LICENSE`），如需商业用途请遵循其许可要求或获取 [Ultralytics 企业许可](https://ultralytics.com/license)。
