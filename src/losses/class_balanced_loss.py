import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedCrossEntropyLoss(nn.Module):
    """
    Class-Balanced Loss dựa trên số mẫu hiệu dụng (Effective Number of Samples)
    Tham khảo: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
    
    Công thức trọng số cho mỗi lớp i:
        W_i = (1 - beta) / (1 - beta^(N_i))
    Trong đó:
        - N_i là số lượng mẫu của lớp i trong tập huấn luyện
        - beta là siêu tham số (thường chọn 0.99, 0.999, 0.9999)
    """
    def __init__(self, samples_per_cls, beta=0.999):
        super(ClassBalancedCrossEntropyLoss, self).__init__()
        
        # Tính số mẫu hiệu dụng: E_n = (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        
        # Chuẩn hóa trọng số sao cho tổng trọng số bằng số lớp
        weights = weights / np.sum(weights) * len(samples_per_cls)
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        device = logits.device
        weights = self.class_weights.to(device)
        return F.cross_entropy(logits, targets, weight=weights)
