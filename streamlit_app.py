import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="自己训练的yolo26n的推理测试", page_icon="🚀", layout="wide")

MODEL_PATH = "runs/detect/train-2/weights/best.pt"
DEFAULT_CONFIDENCE = 0.25


def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"模型文件不存在: {MODEL_PATH}")
        return None
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None


def run_inference(model, image, confidence):
    try:
        results = model.predict(
            source=image,
            conf=confidence,
            save=False,
            verbose=False
        )
        return results[0]
    except Exception as e:
        st.error(f"推理失败: {str(e)}")
        return None


def draw_boxes(image, result):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = result.names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.title("🚀 YOLO26n目标检测推理测试")

with st.sidebar:
    st.header("模型配置")

    confidence = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE,
        step=0.05
    )

    st.header("示例图片")
    sample_images = [f for f in os.listdir("VisDrone_split/images/val/") if f.endswith(('.jpg', '.png', '.jpeg'))]
    selected_sample = st.selectbox("选择示例图片", [""] + sample_images)

col1, col2 = st.columns(2)

with col1:
    st.subheader("输入图片")
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_column_width=True)
    elif selected_sample:
        sample_path = f"VisDrone_split/images/val/{selected_sample}"
        image = Image.open(sample_path)
        st.image(image, caption="示例图片", use_column_width=True)
    else:
        st.info("请上传图片或选择示例图片")

with col2:
    st.subheader("推理结果")

    if (uploaded_file is not None or selected_sample) and st.button("开始推理"):
        with st.spinner("正在加载模型..."):
            model = load_model()

        if model:
            with st.spinner("正在推理..."):
                if uploaded_file:
                    img = Image.open(uploaded_file)
                else:
                    img = Image.open(f"VisDrone_split/images/val/{selected_sample}")

                result = run_inference(model, img, confidence)

                if result:
                    output_img = draw_boxes(img, result)
                    st.image(output_img, caption="检测结果", use_column_width=True)

                    st.subheader("检测信息")
                    for i, box in enumerate(result.boxes):
                        label = result.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        st.write(f"🎯 目标 {i + 1}: {label} (置信度: {conf:.2f})")

                    st.success(f"检测完成！共识别到 {len(result.boxes)} 个目标")