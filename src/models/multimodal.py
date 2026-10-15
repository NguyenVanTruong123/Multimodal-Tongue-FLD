import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class MultimodalFattyLiverModel(nn.Module):
    """
    Mô hình Đa phương thức hợp nhất Ảnh lưỡi + 8 Chỉ số sinh lý:
    - Ảnh: Trích xuất bằng ConvNeXt-Tiny (đặc trưng 768 chiều)
    - Chỉ số: Trích xuất bằng MLP (đặc trưng 64 chiều)
    - Hợp nhất: Nối (concat) -> Fusion Layers (biểu diễn chung)
    - Hai đầu ra:
        1. Đầu ra phân loại đa lớp (3 classes) -> dùng cho Cross-Entropy / Class-Balanced CE
        2. Đầu ra học thứ tự CORAL (2 binary tasks cho 3 mức độ) -> dùng cho CORAL Loss
    Phục vụ cho các cấu hình M3, M4, M5, M6.
    """
    def __init__(self, num_classes=3, tabular_dim=8, pretrained=True, dropout_rate=0.3):
        super(MultimodalFattyLiverModel, self).__init__()
        
        # 1. Nhánh trích xuất đặc trưng ảnh (ConvNeXt-Tiny)
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base_conv = convnext_tiny(weights=weights)
        self.image_feature_dim = base_conv.classifier[2].in_features  # 768
        # Giữ lại các tầng trừ classifier cuối
        self.image_backbone = nn.Sequential(
            base_conv.features,
            base_conv.avgpool
        )
        
        # 2. Nhánh trích xuất đặc trưng 8 chỉ số sinh lý (Tabular MLP)
        self.tabular_feature_dim = 64
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.tabular_feature_dim),
            nn.BatchNorm1d(self.tabular_feature_dim),
            nn.ReLU()
        )
        
        # 3. Tầng hợp nhất đặc trưng (Fusion Layers)
        total_feature_dim = self.image_feature_dim + self.tabular_feature_dim
        self.fusion_dim = 128
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 4. Đầu ra 1: Phân loại 3 lớp (3 classes)
        self.classifier_head = nn.Linear(self.fusion_dim, num_classes)
        
        # 5. Đầu ra 2: Học thứ tự CORAL (K - 1 = 2 ngưỡng cho 3 mức độ bệnh)
        self.coral_head = nn.Linear(self.fusion_dim, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, image, tabular):
        # Trích xuất ảnh -> Flatten thành vector
        img_feats = self.image_backbone(image)
        img_feats = torch.flatten(img_feats, 1)
        
        # Trích xuất chỉ số sinh lý
        tab_feats = self.tabular_mlp(tabular)
        
        # Nối hai nguồn đặc trưng
        fused = torch.cat([img_feats, tab_feats], dim=1)
        shared_repr = self.fusion(fused)
        
        # Đầu ra 1: Logits 3 lớp
        multiclass_logits = self.classifier_head(shared_repr)
        
        # Đầu ra 2: Logits học thứ tự CORAL
        # y > 0 và y > 1
        coral_val = self.coral_head(shared_repr)
        coral_logits = coral_val + self.coral_bias
        
        return multiclass_logits, coral_logits
