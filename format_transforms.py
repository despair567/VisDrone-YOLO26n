import os
import cv2

# 动态获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "VisDrone_split")

# VisDrone 类别映射
CLASS_MAP = {
    0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
}


def convert_one_label(txt_path, img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[警告] 无法读取图片: {img_path}")
        return
    h, w = img.shape[:2]

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 8:
            continue

        x1, y1, bw, bh = map(float, parts[:4])
        cls_id = int(float(parts[5]))

        if cls_id == 0 or cls_id not in CLASS_MAP:
            continue

        new_cls = CLASS_MAP[cls_id]
        x_center = (x1 + bw / 2.0) / w
        y_center = (y1 + bh / 2.0) / h
        bw /= w
        bh /= h

        # 裁剪越界框
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)

        yolo_lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines))
        if yolo_lines:
            f.write('\n')


def main():
    print("开始进行 YOLO 格式转换（就地转换）...")
    splits = ["train", "val"]

    for split in splits:
        img_dir = os.path.join(DATASET_ROOT, "images", split)
        label_dir = os.path.join(DATASET_ROOT, "labels", split)

        if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
            print(f"[跳过] {split} 目录不存在")
            continue

        ann_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        print(f"\n================ 开始处理 {split} ({len(ann_files)} 个标签) ================")

        success_count = 0
        for ann_name in ann_files:
            base = os.path.splitext(ann_name)[0]
            txt_path = os.path.join(label_dir, ann_name)

            # 寻找对应的图片
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                tmp_path = os.path.join(img_dir, base + ext)
                if os.path.exists(tmp_path):
                    img_path = tmp_path
                    break

            if img_path is None:
                continue

            # 就地覆盖转换
            convert_one_label(txt_path, img_path, txt_path)
            success_count += 1

        print(f"✅ {split} 转换完成！成功覆盖 {success_count} 个文件。")

    print("\n全部转换结束！")


if __name__ == "__main__":
    main()