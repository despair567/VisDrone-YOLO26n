import os
import shutil
import random

# 数据集的原始路径
Raw_Images_Dir = "dataset/VisDrone2019-DET-train/images" #图片目录
Raw_Labels_Dir = "dataset/VisDrone2019-DET-train/annotations" #标签目录

# 输出划分后的数据集路径
Output_Dir = r"./VisDrone_split"

# 划分比例
Train_Ratio = 0.8
Val_Ratio = 0.1
Test_Ratio = 0.1

# 固定随机种子，保证每次划分结果一样、
random.seed(42)

def create_dirs(base_dir):
    subsets = ["train", "val"]
    for subset in subsets:
        os.makedirs(os.path.join(base_dir,"images",subset), exist_ok=True)
        os.makedirs(os.path.join(base_dir,"labels",subset), exist_ok=True)

def get_all_image_files(img_dir):
    # 返回完整文件名，不去扩展名
    return [f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


def split_dataset(file_list,train_ratio,val_ratio,test_ratio):
    random.shuffle(file_list)
    total = len(file_list)
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)

    train_files = file_list[:train_num]
    val_files = file_list[train_num:train_num+val_num]
    test_files = file_list[train_num+val_num:]
    return train_files, val_files, test_files

def copy_files(file_list, img_src_dir, label_src_dir, img_dst_dir, label_dst_dir):
    for filename in file_list:
        # 图片：filename
        img_src = os.path.join(img_src_dir, filename)
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(img_dst_dir, filename))
        # 标注：去掉扩展名再拼 .txt
        stem = os.path.splitext(filename)[0]
        label_src = os.path.join(label_src_dir, stem + '.txt')
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(label_dst_dir, stem + '.txt'))

if __name__ == "__main__":
    # 创建输出目录
    create_dirs(Output_Dir)

    # 获取所有图片名
    all_files = get_all_image_files(Raw_Images_Dir)
    print(f"总图片数量：{len(all_files)}")

    #划分
    train_files, val_files, test_files = split_dataset(all_files, Train_Ratio, Val_Ratio, Test_Ratio)
    print(f"训练集数量：{len(train_files)}")
    print(f"验证集数量：{len(val_files)}")

    # 复制文件
    copy_files(train_files,Raw_Images_Dir,Raw_Labels_Dir,
               os.path.join(Output_Dir,"images","train"),
               os.path.join(Output_Dir,"labels","train"))
    copy_files(val_files,Raw_Images_Dir,Raw_Labels_Dir,
               os.path.join(Output_Dir,"images","val"),
               os.path.join(Output_Dir,"labels","val"))

    print("数据集划分完成！")