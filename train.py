from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO('yolo26n.pt')
    # 开始训练

    results = model.train(
        data="Visdrone.yaml",      # 配置文件
        epochs=10,                 # 轮数
        imgsz=640,                 # 图片大小
        batch=4,                   # 批次大小
        device=0,                  # 使用GPU
        workers=0                  # 关键：windows 必须设 workers=0！
    )