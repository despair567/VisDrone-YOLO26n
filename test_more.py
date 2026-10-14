from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train/weights/best.pt")

    img_path = "test_images/"

    model.predict(
        source=img_path,
        save = True, #保存画框后的图片
        conf = 0.25, #置信度阈值
        batch = 4    #批次大小
    )