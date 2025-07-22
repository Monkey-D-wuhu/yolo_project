import json
import os
from PIL import Image
from tqdm import tqdm

def process_annotations(json_file, images_folder, labels_folder):
    # 读取 JSON 文件
    with open(json_file, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']
    categories = data['categories']

    # 构建 category_id → class_id 映射（连续编号）
    cat_id_map = {cat['id']: idx for idx, cat in enumerate(categories)}

    converted = 0  # 统计写入的框数
    skipped = 0    # 跳过的图像数

    for annotation in tqdm(annotations, desc=f"处理 {json_file}"):
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        class_id = cat_id_map[category_id]  # 正确转换

        # 获取图片路径
        image_filename = str(image_id).zfill(12) + '.jpg'
        file_path = os.path.join(images_folder, image_filename)
        if not os.path.exists(file_path):
            skipped += 1
            continue

        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except:
            skipped += 1
            continue

        # 获取 bbox 并归一化
        bbox = annotation['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        w_norm = bbox[2] / width
        h_norm = bbox[3] / height

        # YOLO标签内容
        label_content = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        label_file_path = os.path.join(labels_folder, str(image_id).zfill(12) + '.txt')
        with open(label_file_path, 'a') as file:
            file.write(label_content + '\n')
        converted += 1

    print(f"✅ 处理完成：共写入 {converted} 个目标框，跳过 {skipped} 个图像。")

def create_labels_folder():
    labels_folder = 'labels'
    train_labels_folder = os.path.join(labels_folder, 'train2017')
    val_labels_folder = os.path.join(labels_folder, 'val2017')
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    return train_labels_folder, val_labels_folder

def main():
    train_labels_folder, val_labels_folder = create_labels_folder()
    process_annotations('annotations_trainval2017/annotations/instances_train2017.json', 'images/train2017', train_labels_folder)
    process_annotations('annotations_trainval2017/annotations/instances_val2017.json', 'images/val2017', val_labels_folder)
    print("🎉 所有标签已成功转换为 YOLO 格式！")

if __name__ == "__main__":
    main()
