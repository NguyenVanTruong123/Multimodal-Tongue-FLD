import torch
import torch.nn as nn

class M1_MLP(nn.Module):
    """
    Mô hình M1: Chỉ sử dụng 8 chỉ số sinh lý (Tabular data)
    Kiến trúc Multi-Layer Perceptron (MLP) với BatchNorm, ReLU và Dropout.
    """
    def __init__(self, input_dim=8, num_classes=3, hidden_dims=[64, 32], dropout_rate=0.3):
        super(M1_MLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
