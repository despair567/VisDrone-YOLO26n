1. 主要功能
数据集划分 (data_split.py)：自动将 VisDrone 原始数据集按 8:1:1 的比例划分为训练集、验证集和测试集（代码目前实际复制了 train 和 val，test 部分未复制），并按 YOLO 结构生成对应目录。
/n
标签格式转换 (format_transforms.py)：将 VisDrone 特有的标注格式（x1, y1, bw, bh, score, class_id）就地转换为 YOLO 格式（class_id x_center y_center w h），并进行类别映射和越界坐标裁剪。
/n
模型训练 (train.py)：基于 yolo26n.pt 预训练权重，加载 Visdrone.yaml 配置文件进行微调训练，默认训练 10 轮，批次大小 4。
/n
单张/批量推理 (test_one.py, test_more.py)：支持对单张图片或整个文件夹的图片进行推理，并保存带画框的推理结果。
/n
Web 可视化系统 (streamlit_app.py)：基于 Streamlit 搭建的交互式网页端，支持上传图片或选择验证集中的示例图片，动态调整置信度阈值，并实时展示检测结果、画框图像和详细的目标信息。
/n
2. 安装方法
/n
2.1 环境要求
/n
Python 3.8 及以上版本

推荐使用 Anaconda 创建虚拟环境

支持 GPU (CUDA) 环境以加速训练和推理

2.2 安装依赖
在项目根目录下，运行以下命令安装所需的 Python 库：
/n
bash
pip install ultralytics streamlit opencv-python numpy pillow
/n
2.3 准备数据集
下载 VisDrone2019-DET 数据集。
/n
将数据集解压并放置在项目根目录的 dataset/ 文件夹下，确保目录结构为：

text
dataset/
└── VisDrone2019-DET-train/
    ├── images/          # 存放 .jpg 图片
    └── annotations/     # 存放 .txt 原始标签
3. 使用方法
步骤一：数据预处理
运行 data_split.py 对原始数据进行划分，生成 VisDrone_split 目录：

bash
python data_split.py
运行 format_transforms.py 进行 YOLO 格式标签转换：

bash
python format_transforms.py
⚠️ 注意：此脚本为就地覆盖转换（In-place conversion），会直接修改 VisDrone_split 下的标签文件。如果你需要保留原始 VisDrone 格式的标签，请提前备份。

步骤二：模型训练
确保项目根目录下已有 Visdrone.yaml 配置文件（用于指定数据集路径和类别数），然后运行：

bash
python train.py
注：训练结束后，权重文件通常保存在 runs/detect/train/weights/best.pt。

步骤三：模型测试与推理
单张图片测试：将待测图片命名为 test.jpg 放在根目录，运行：

bash
python test_one.py
批量图片测试：将待测图片放入 test_images/ 文件夹，运行：

bash
python test_more.py
步骤四：启动 Web 可视化系统
运行以下命令启动 Streamlit 应用：

bash
streamlit run streamlit_app.py
启动后，浏览器会自动打开页面（默认地址为 http://localhost:8501）。您可以在左侧侧边栏调整置信度阈值，在左侧选择示例图片或上传本地图片，点击“开始推理”即可在右侧查看结果。

4. 输入输出示例
4.1 输入示例
图像输入：无人机航拍的高分辨率图像（如 .jpg, .jpeg, .png 格式）。支持单张上传（test.jpg）或选择验证集图片（VisDrone_split/images/val/）。

参数输入：置信度阈值（Confidence Threshold），默认为 0.25，可在 Streamlit 侧边栏通过滑块在 0.0 - 1.0 之间调整。

4.2 输出示例
图像输出：在原图上绘制绿色边界框（Bounding Box），并在框上方标注类别名称和置信度（如 car: 0.85）。

控制台/终端输出（以 test_one.py 为例）：

text
image 1/1 /path/to/test.jpg: 640x640 3 cars, 1 pedestrian, 2 trucks, 45.2ms
Speed: 1.5ms preprocess, 45.2ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 640)
Results saved to runs/detect/predict
Web 界面输出：

右侧展示带有检测框的图片。

下方以列表形式展示检测信息：

text
🎯 目标 1: car (置信度: 0.89)
🎯 目标 2: pedestrian (置信度: 0.76)
🎯 目标 3: truck (置信度: 0.65)
底部显示成功提示：检测完成！共识别到 3 个目标。
