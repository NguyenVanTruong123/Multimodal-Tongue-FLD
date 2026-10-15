import torch
import torch.nn as nn
import torch.nn.functional as F

class CoralLoss(nn.Module):
    """
    CORAL Loss (Consistent Rank Logits for Ordinal Regression)
    Tham khảo: Cao et al., "Rank consistent ordinal regression for neural networks...", Pattern Recognition Letters 2020.
    
    Với bài toán 3 mức độ (K=3):
        - Nhãn 0: [0, 0]
        - Nhãn 1: [1, 0]
        - Nhãn 2: [1, 1]
    CORAL tính tổng Binary Cross Entropy trên từng ngưỡng thứ tự.
    """
    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, coral_logits, targets):
        """
        coral_logits: Tensor kích thước (batch_size, num_classes - 1) = (batch_size, 2)
        targets: Tensor kích thước (batch_size,) chứa nhãn 0, 1, 2
        """
        num_classes = coral_logits.size(1) + 1
        
        # Chuyển đổi nhãn phân loại (0, 1, 2) sang nhãn nhị phân thứ tự
        # Ví dụ target=1 -> [1, 0]; target=2 -> [1, 1]; target=0 -> [0, 0]
        levels = torch.arange(num_classes - 1, device=targets.device).expand(len(targets), -1)
        binary_targets = (targets.unsqueeze(1) > levels).float()
        
        # Binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(coral_logits, binary_targets)
        return loss

def coral_predict(coral_logits):
    """
    Dự đoán nhãn từ logits của CORAL:
    Đếm số ngưỡng mà xác suất Sigmoid(logit) > 0.5 (hoặc logit > 0)
    """
    probabilities = torch.sigmoid(coral_logits)
    predictions = torch.sum(probabilities > 0.5, dim=1)
    return predictions
