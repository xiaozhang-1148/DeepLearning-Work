from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .vocab import vocab

# --- 新增：自定义变换类，用于保持纵横比缩放和填充 ---
class ResizeAndPad:
    def __init__(self, output_size: int, fill: int = 255):
        """
        Args:
            output_size (int): 目标正方形尺寸.
            fill (int): 填充区域的像素值 (0=黑色, 255=白色).
        """
        self.output_size = output_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. 计算缩放后的尺寸
        w, h = img.size
        if w > h:
            new_w = self.output_size
            new_h = int(h * (self.output_size / w))
        else:
            new_h = self.output_size
            new_w = int(w * (self.output_size / h))
        
        # 2. 等比例缩放图像
        resized_img = img.resize((new_w, new_h), Image.LANCZOS)

        # 3. 创建一个空白画布
        new_img = Image.new("L", (self.output_size, self.output_size), self.fill)
        
        # 4. 将缩放后的图像粘贴到画布中心
        paste_x = (self.output_size - new_w) // 2
        paste_y = (self.output_size - new_h) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        
        return new_img

# 1. 定义一个标准的 Dataset 类
class CROHMEDataset(Dataset):
    def __init__(self, data: List[Tuple[str, Image.Image, List[str]]], is_train: bool, scale_aug: bool):
        super().__init__()
        self.data = data
        self.is_train = is_train
        
        transform_list = []
        if is_train and scale_aug:
            transform_list.append(transforms.RandomApply([
                transforms.Resize((int(256 * 0.8), int(256 * 0.8))),
            ], p=0.5))
        
        transform_list.extend([
            ResizeAndPad(256, fill=255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, FloatTensor, List[int]]:
        img_name, img, formula_tokens = self.data[idx]
        
        # 应用图像变换
        img_tensor = self.transform(img)
        
        # 将 formula tokens 转换为 indices
        indices = vocab.words2indices(formula_tokens)
        
        return img_name, img_tensor, indices

# 2. 定义 Batch 类
@dataclass
class Batch:
    img_bases: List[str]
    imgs: FloatTensor
    mask: LongTensor
    indices: List[List[int]]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch_data: List[Tuple[str, FloatTensor, List[int]]]) -> Batch:
    # batch_data 是一个元组列表: [(img_name, img_tensor, indices), ...]
    img_names, img_tensors, indices_list = zip(*batch_data)

    # 将图像张量堆叠成一个批次
    imgs = torch.stack(img_tensors, 0)
    
    b, _, h, w = imgs.shape
    mask = torch.zeros((b, h, w), dtype=torch.bool)

    return Batch(
        img_bases=list(img_names),
        imgs=imgs,
        mask=mask,
        indices=list(indices_list)
    )

# 第一步 : extract_data
def extract_data(archive: ZipFile, dir_name: str) -> List[Tuple[str, Image.Image, List[str]]]:
    """从 zip 中提取数据，返回 (图片名, PIL Image, formula tokens) 列表"""
    data = []
    caption_path = f"data/{dir_name}/caption.txt"
    try:
        with archive.open(caption_path, "r") as f:
            captions = f.read().decode('utf-8').strip().splitlines()
    except KeyError:
        print(f"Error: Cannot find {caption_path} in the zip file.")
        return []

    for line in captions:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        img_name, formula = parts[0], parts[1]
        
        img_path = f"data/{dir_name}/img/{img_name}"
        try:
            with archive.open(img_path, "r") as f:
                img = Image.open(f).convert('L').copy()
                data.append((img_name, img, formula))
        except KeyError:
            print(f"Warning: Cannot find image {img_path} for caption line: {line}")

    print(f"Extracted {len(data)} samples from: {dir_name}")
    return data


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str,
        train_batch_size: int = 32,
        eval_batch_size: int = 24,
        num_workers: int = 8,
        scale_aug: bool = True,
    ) -> None:
        super().__init__()
        self.zipfile_path = zipfile_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug
        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage in ("fit", None):
                train_data = extract_data(archive, "train")
                val_data = extract_data(archive, "val")
                self.train_dataset = CROHMEDataset(train_data, is_train=True, scale_aug=self.scale_aug)
                self.val_dataset = CROHMEDataset(val_data, is_train=False, scale_aug=False)
            if stage in ("test", None):
                test_data = extract_data(archive, "test")
                self.test_dataset = CROHMEDataset(test_data, is_train=False, scale_aug=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
