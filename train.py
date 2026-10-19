from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26-p2.yaml").load("yolo26n.pt")

    model.train(
        data="VisDrone.yaml",
        epochs=100,
        warmup_epochs=5.0,
        close_mosaic=10,
        imgsz=640,
        batch=4,
        optimizer="MuSGD",
        lr0=0.025,
        lrf=0.01,
        cos_lr=True,
        patience=30,
        device=0,
        project="runs_visdrone",
        name="yolo26n_p2",
        exist_ok=True,
        pretrained=True,
        amp=True,
        workers=4,
        cache="ram",
        max_det=1000,
        plots=True,
    )
