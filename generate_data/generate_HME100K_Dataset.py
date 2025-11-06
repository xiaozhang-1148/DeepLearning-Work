import os
import json
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

def create_dir(path):
    """创建目录，如果存在则先删除再创建"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_captions(caption_file_path):
    """加载标注文件到字典中"""
    captions = {}
    with open(caption_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                img_name, caption = line.strip().split('\t', 1)
                captions[img_name] = caption
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")
    return captions

def save_captions(captions_dict, file_path):
    """将标注字典保存到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for img_name, caption in captions_dict.items():
            f.write(f"{img_name}\t{caption}\n")

def copy_files_and_captions(file_list, src_img_dir, dest_img_dir, all_captions, dest_caption_path):
    """复制图片并生成对应的标注文件"""
    subset_captions = {}
    create_dir(dest_img_dir)
    
    for filename in tqdm(file_list, desc=f"Copying to {os.path.basename(dest_img_dir)}"):
        src_path = os.path.join(src_img_dir, filename)
        dest_path = os.path.join(dest_img_dir, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            if filename in all_captions:
                subset_captions[filename] = all_captions[filename]
        else:
            print(f"Warning: Source image not found and skipped: {src_path}")
            
    save_captions(subset_captions, dest_caption_path)

def main(args):
    subset_dir = args.subset_dir
    source_image_dir = args.image_dir
    source_caption_file = args.caption_file

    # 定义输出路径
    output_base_dir = os.path.join(args.output_dir, 'data')
    test_dir = os.path.join(output_base_dir, 'test')
    train_dir = os.path.join(output_base_dir, 'train')
    val_dir = os.path.join(output_base_dir, 'val')

    # 打印路径配置
    print("--- Path Configuration ---")
    print(f"Source Subset JSONs Path: {subset_dir}")
    print(f"Source Images Path:       {source_image_dir}")
    print(f"Source Captions Path:     {source_caption_file}")
    print("-" * 26)
    print(f"Output Base Path:         {output_base_dir}")
    print("-" * 26)

    # 1. 加载所有标注 (提前加载，后续所有部分都会用到)
    print("Loading all captions...")
    all_captions = load_captions(source_caption_file)
    print(f"Loaded {len(all_captions)} captions.")

    # 2. 创建测试集，并按难度划分
    print("\nCreating test set (subdivided by difficulty)...")
    test_filenames_set = set()
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"  Processing '{difficulty}' subset...")
        
        # 为当前难度定义输出路径
        difficulty_test_dir = os.path.join(test_dir, difficulty)
        difficulty_img_dir = os.path.join(difficulty_test_dir, 'img')
        difficulty_caption_path = os.path.join(difficulty_test_dir, 'caption.txt')

        # 从对应的json加载文件名
        json_path = os.path.join(subset_dir, f'{difficulty}.json')
        with open(json_path, 'r') as f:
            difficulty_filenames = [f"{name}.jpg" for name in json.load(f)[:3000]]
        
        # 复制文件和标注
        copy_files_and_captions(
            difficulty_filenames,
            source_image_dir,
            difficulty_img_dir,
            all_captions,
            difficulty_caption_path
        )
        
        # 将文件名添加到总的测试集set中，以便后续排除
        test_filenames_set.update(difficulty_filenames)

    print(f"\nFound {len(test_filenames_set)} unique files for the total test set.")

    # 3. 准备训练集和验证集文件
    print("\nPreparing files for training and validation sets...")
    all_source_files = set(os.listdir(source_image_dir))
    remaining_files = sorted(list(all_source_files - test_filenames_set))
    print(f"Total files: {len(all_source_files)}")
    print(f"Test files: {len(test_filenames_set)}")
    print(f"Remaining files for train/val: {len(remaining_files)}")

    # 4. 划分训练集和验证集
    train_filenames, val_filenames = train_test_split(
        remaining_files,
        test_size=0.1,
        random_state=42  # 使用固定的随机种子
    )
    print(f"Splitting into {len(train_filenames)} training files and {len(val_filenames)} validation files.")

    # 5. 创建训练集
    print("\nCreating training set...")
    copy_files_and_captions(
        train_filenames,
        source_image_dir,
        os.path.join(train_dir, 'img'),
        all_captions,
        os.path.join(train_dir, 'caption.txt')
    )

    # 6. 创建验证集
    print("\nCreating validation set...")
    copy_files_and_captions(
        val_filenames,
        source_image_dir,
        os.path.join(val_dir, 'img'),
        all_captions,
        os.path.join(val_dir, 'caption.txt')
    )

    print("\nDataset generation complete!")
    print(f"Data has been saved to: {output_base_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate HME100K dataset splits from specified source paths.")
    
    parser.add_argument(
        '--subset_dir', 
        type=str, 
        required=True, 
        help="Path to the directory containing easy.json, medium.json, and hard.json."
    )
    parser.add_argument(
        '--image_dir', 
        type=str, 
        required=True, 
        help="Path to the directory containing all source images."
    )
    parser.add_argument(
        '--caption_file', 
        type=str, 
        required=True, 
        help="Path to the main caption.txt file."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="Path to the directory where the 'data' folder will be created."
    )
    
    args = parser.parse_args()
    main(args)