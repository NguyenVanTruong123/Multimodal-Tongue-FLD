import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class TongueFLDDataset(Dataset):
    """
    Dataset class cho bộ dữ liệu Tongue-FLD:
    Hỗ trợ nạp cả ảnh lưỡi và 8 chỉ số sinh lý hoặc từng phần riêng biệt.
    """
    def __init__(self, df, image_dir=None, feature_cols=None, transform=None, mode='multimodal'):
        """
        - df: DataFrame pandas đã lọc theo fold
        - image_dir: Thư mục chứa file ảnh lưỡi
        - feature_cols: Danh sách tên 8 cột chỉ số sinh lý
        - transform: Các phép biến đổi ảnh (torchvision transforms)
        - mode: 'tabular_only' (M1), 'image_only' (M2), 'multimodal' (M3-M6)
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.feature_cols = feature_cols
        self.transform = transform
        self.mode = mode
        
        # Nhãn mức độ bệnh (0: Không mắc, 1: Nhẹ, 2: Trung bình/Nặng)
        self.labels = self.df['label'].values.astype(np.int64)
        
        # 8 chỉ số sinh lý
        if self.feature_cols is not None:
            self.tabular_features = self.df[self.feature_cols].values.astype(np.float32)
        else:
            self.tabular_features = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # 1. Chế độ chỉ dùng bảng chỉ số (M1)
        if self.mode == 'tabular_only':
            tab = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
            return tab, label

        # 2. Đọc ảnh lưỡi (cho M2 và multimodal)
        img_name = self.df.iloc[idx]['image_name']
        img_path = os.path.join(self.image_dir, img_name) if self.image_dir else img_name
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Chế độ chỉ dùng ảnh (M2)
        if self.mode == 'image_only':
            return image, label

        # Chế độ đa phương thức (M3 -> M6)
        tab = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        return image, tab, label
