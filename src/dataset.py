import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import scipy.ndimage


class TransAttUnetDataset(Dataset):
    def __init__(self, data_dir, split_file, mode='train', final_size=(128, 128)):
        """
        Dataset loader cho TransAttUnet
        data_dir: thư mục processed_data (images/ + masks/)
        split_file: file json chia train/val/test
        mode: 'train' hoặc 'val'
        final_size: kích thước crop/rescale ảnh ROI
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.mode = mode
        self.final_size = final_size

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Không tìm thấy file split: {split_file}")

        with open(split_file, 'r') as f:
            splits = json.load(f)

        self.file_list = splits.get(mode, [])

        if not self.file_list:
            print(f"Cảnh báo: Tập dữ liệu '{mode}' rỗng!")
        else:
            print(f"Đã load tập '{mode}': {len(self.file_list)} mẫu.")

    def __len__(self):
        return len(self.file_list)

    def augment(self, image, mask):
        """
        Hàm thực hiện Augmentation thủ công bằng Numpy.
        """
        # Random Flip Horizontal (Lật ngang)
        if random.random() > 0.5:
            image = np.flip(image, axis=1)  # Axis 1 là chiều rộng
            mask = np.flip(mask, axis=1)

        # Random Flip Vertical (Lật dọc)
        if random.random() > 0.5:
            image = np.flip(image, axis=0)  # Axis 0 là chiều cao
            mask = np.flip(mask, axis=0)

        # Random Rotation (Xoay 90, 180, 270 độ)
        k = random.randint(0, 3)  # 0: 0 độ, 1: 90 độ, 2: 180 độ, 3: 270 độ
        if k > 0:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        return image, mask

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # Load file .npy
        # Image shape: (H, W)
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)

        # --- DATA AUGMENTATION --- chỉ áp dụng cho train
        if self.mode == 'train':
            image, mask = self.augment(image, mask)
        # ---------------------------------

        # ----- CROP và RESIZE về final_size -----
        h, w = image.shape
        target_h, target_w = self.final_size

        # Nếu ảnh nhỏ hơn target, padding
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # Resize về target_size
        image = scipy.ndimage.zoom(
            image,
            (target_h / image.shape[0], target_w / image.shape[1]),
            order=1  # bilinear
        )
        mask = scipy.ndimage.zoom(
            mask,
            (target_h / mask.shape[0], target_w / mask.shape[1]),
            order=0  # nearest
        )

        # Thêm chiều Channels (1, H, W)
        image = np.expand_dims(image.copy(), axis=0)
        mask = np.expand_dims(mask.copy(), axis=0)

        # Chuyển sang Tensor
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor