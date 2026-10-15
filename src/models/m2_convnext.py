import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class M2_ConvNeXt(nn.Module):
    """
    Mô hình M2: Chỉ sử dụng ảnh lưỡi (Tongue images)
    Sử dụng ConvNeXt-Tiny pre-trained trích xuất đặc trưng ảnh.
    """
    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.3):
        super(M2_ConvNeXt, self).__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        
        # Lấy kích thước vector đầu ra của backbone ConvNeXt-Tiny (thường là 768)
        in_features = self.backbone.classifier[2].in_features
        
        # Thay thế lớp classifier cuối bằng classifier 3 lớp của bài toán
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
