import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_absolute_error, mean_squared_error, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Tính toán các chỉ số đánh giá theo đúng đề cương NCKH:
    - Accuracy
    - Macro-F1 (trung bình F1 giữa các lớp)
    - Recall từng lớp (đặc biệt theo dõi Recall lớp 2 - trung bình/nặng)
    - MAE, RMSE (đánh giá mức độ sai lệch thứ tự bệnh)
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Recall theo từng mức độ bệnh: 0 (Không mắc), 1 (Nhẹ), 2 (Trung bình/Nặng)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'recall_class_0': recalls[0] if len(recalls) > 0 else 0.0,
        'recall_class_1': recalls[1] if len(recalls) > 1 else 0.0,
        'recall_class_2': recalls[2] if len(recalls) > 2 else 0.0,
        'mae': mae,
        'rmse': rmse,
        'confusion_matrix': cm
    }
