"""
Kịch bản huấn luyện (Training Engine) cho đề tài NCKH:
Phân loại mức độ gan nhiễm mỡ đa phương thức (Tongue-FLD).
Hỗ trợ 6 cấu hình mô hình thực nghiệm (Ablation Study):
- M1: Chỉ dùng 8 chỉ số sinh lý (MLP)
- M2: Chỉ dùng ảnh lưỡi (ConvNeXt-Tiny)
- M3: Multimodal Baseline (ConvNeXt-Tiny + MLP + Cross-Entropy)
- M4: M3 + Class-Balanced Cross-Entropy Loss
- M5: M3 + Mô hình hai đầu ra (Cross-Entropy + CORAL Loss)
- M6: M3 + Mô hình hai đầu ra (Class-Balanced Loss + CORAL Loss) - Đề xuất
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Nạp các module trong src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import TongueFLDDataset
from src.models.m1_mlp import M1_MLP
from src.models.m2_convnext import M2_ConvNeXt
from src.models.multimodal import MultimodalFattyLiverModel
from src.losses.class_balanced_loss import ClassBalancedCrossEntropyLoss
from src.losses.coral_loss import CoralLoss, coral_predict
from src.utils.metrics import compute_metrics


def set_seed(seed=42):
    """Cố định seed để đảm bảo tính tái lập (reproducibility)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms():
    """Khởi tạo pipeline tiền xử lý và tăng cường ảnh"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def prepare_fold_data(df, fold, numeric_cols, feature_cols):
    """
    Chuẩn bị dữ liệu cho Fold hiện tại:
    - Tách tập Train (fold != current_fold) và tập Val (fold == current_fold)
    - Chuẩn hóa Z-Score: Fit scaler CHỈ TRÊN TẬP TRAIN, sau đó transform Val (chống rò rỉ dữ liệu)
    """
    train_df = df[df['fold'] != fold].copy().reset_index(drop=True)
    val_df = df[df['fold'] == fold].copy().reset_index(drop=True)
    
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    
    return train_df, val_df


def build_model(model_name, num_classes=3, tabular_dim=8, dropout_rate=0.3):
    """Khởi tạo kiến trúc mô hình tương ứng"""
    if model_name == 'm1':
        return M1_MLP(input_dim=tabular_dim, num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'm2':
        return M2_ConvNeXt(num_classes=num_classes, pretrained=True, dropout_rate=dropout_rate)
    elif model_name in ['m3', 'm4', 'm5', 'm6']:
        return MultimodalFattyLiverModel(num_classes=num_classes, tabular_dim=tabular_dim, pretrained=True, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")


def get_dataset_mode(model_name):
    """Xác định chế độ nạp dữ liệu cho TongueFLDDataset"""
    if model_name == 'm1':
        return 'tabular_only'
    elif model_name == 'm2':
        return 'image_only'
    else:
        return 'multimodal'


def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_coral, model_name, alpha, device):
    """Vòng lặp huấn luyện trong 1 epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        if model_name == 'm1':
            tabular, labels = batch
            tabular, labels = tabular.to(device), labels.to(device)
            logits = model(tabular)
            loss = criterion_cls(logits, labels)
            
        elif model_name == 'm2':
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion_cls(logits, labels)
            
        elif model_name in ['m3', 'm4']:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            logits, _ = model(images, tabular)
            loss = criterion_cls(logits, labels)
            
        elif model_name in ['m5', 'm6']:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            logits, coral_logits = model(images, tabular)
            loss_cls = criterion_cls(logits, labels)
            loss_coral = criterion_coral(coral_logits, labels)
            loss = alpha * loss_cls + (1.0 - alpha) * loss_coral
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion_cls, criterion_coral, model_name, alpha, device):
    """Vòng lặp đánh giá trên tập Validation"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        if model_name == 'm1':
            tabular, labels = batch
            tabular, labels = tabular.to(device), labels.to(device)
            logits = model(tabular)
            loss = criterion_cls(logits, labels)
            preds = torch.argmax(logits, dim=1)
            
        elif model_name == 'm2':
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion_cls(logits, labels)
            preds = torch.argmax(logits, dim=1)
            
        elif model_name in ['m3', 'm4']:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            logits, _ = model(images, tabular)
            loss = criterion_cls(logits, labels)
            preds = torch.argmax(logits, dim=1)
            
        elif model_name in ['m5', 'm6']:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            logits, coral_logits = model(images, tabular)
            loss_cls = criterion_cls(logits, labels)
            loss_coral = criterion_coral(coral_logits, labels)
            loss = alpha * loss_cls + (1.0 - alpha) * loss_coral
            # Dự đoán kết hợp: dùng argmax logits phân loại
            preds = torch.argmax(logits, dim=1)
            
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics['val_loss'] = total_loss / len(dataloader.dataset)
    return metrics


def train_single_fold(fold, df, args, device):
    """Quy trình huấn luyện độc lập cho 1 Fold cụ thể"""
    print(f"\n{'='*25} BẮT ĐẦU FOLD {fold} ({args.model.upper()}) {'='*25}")
    set_seed(args.seed + fold)
    
    numeric_cols = ['Age', 'Height', 'Waist', 'Hipline', 'Weight', 'SBP', 'DBP']
    feature_cols = ['Gender', 'Age', 'Height', 'Waist', 'Hipline', 'Weight', 'SBP', 'DBP']
    
    train_df, val_df = prepare_fold_data(df, fold, numeric_cols, feature_cols)
    train_transform, val_transform = get_transforms()
    mode = get_dataset_mode(args.model)
    
    train_dataset = TongueFLDDataset(train_df, image_dir=args.images_dir, feature_cols=feature_cols, transform=train_transform, mode=mode)
    val_dataset = TongueFLDDataset(val_df, image_dir=args.images_dir, feature_cols=feature_cols, transform=val_transform, mode=mode)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Khởi tạo mô hình mới tinh (không nhớ kiến thức fold khác)
    model = build_model(args.model, num_classes=3, tabular_dim=len(feature_cols), dropout_rate=args.dropout).to(device)
    
    # Thiết lập hàm mất mát
    criterion_coral = CoralLoss()
    if args.model in ['m1', 'm2', 'm3', 'm5']:
        criterion_cls = nn.CrossEntropyLoss()
    elif args.model in ['m4', 'm6']:
        # Tính số mẫu mỗi lớp trên tập train của fold này
        samples_per_cls = train_df['label'].value_counts().sort_index().values
        criterion_cls = ClassBalancedCrossEntropyLoss(samples_per_cls=samples_per_cls, beta=args.cb_beta)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_macro_f1 = -1.0
    best_metrics = None
    best_epoch = -1
    no_improve_epochs = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{args.model}_fold_{fold}.pth")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_coral, args.model, args.alpha, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion_cls, criterion_coral, args.model, args.alpha, device)
        
        current_f1 = val_metrics['macro_f1']
        
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_metrics = val_metrics
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            mark = "(* Lưu trọng số tốt nhất)"
        else:
            no_improve_epochs += 1
            mark = ""
            
        if epoch % 5 == 0 or epoch == 1 or mark != "":
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val Macro-F1: {current_f1*100:.2f}% | MAE: {val_metrics['mae']:.4f} {mark}")
            
        # Early stopping
        if no_improve_epochs >= args.patience:
            print(f"--> Kích hoạt Early Stopping tại Epoch {epoch} (không cải thiện sau {args.patience} epochs).")
            break
            
    print(f"--> KẾT THÚC FOLD {fold}: Đạt Macro-F1 đỉnh cao tại Epoch {best_epoch}: {best_macro_f1*100:.2f}%")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình NCKH phân loại gan nhiễm mỡ Tongue-FLD")
    parser.add_argument('--model', type=str, default='m1', choices=['m1', 'm2', 'm3', 'm4', 'm5', 'm6'], help="Cấu hình mô hình cần huấn luyện")
    parser.add_argument('--fold', type=int, default=0, help="Fold cần chạy (0-4), hoặc -1 để chạy toàn bộ 5 folds")
    parser.add_argument('--epochs', type=int, default=50, help="Số lượng epoch huấn luyện")
    parser.add_argument('--batch_size', type=int, default=64, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=1e-3, help="Tốc độ học (Learning Rate)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Hệ số Weight Decay")
    parser.add_argument('--dropout', type=float, default=0.3, help="Tỷ lệ Dropout")
    parser.add_argument('--patience', type=int, default=15, help="Số epoch chờ cho Early Stopping")
    parser.add_argument('--alpha', type=float, default=0.5, help="Trọng số kết hợp loss cho M5, M6: alpha * CE + (1 - alpha) * CORAL")
    parser.add_argument('--cb_beta', type=float, default=0.999, help="Hệ số beta cho Class-Balanced Loss")
    parser.add_argument('--seed', type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument('--num_workers', type=int, default=0, help="Số worker cho DataLoader (Windows nên để 0)")
    parser.add_argument('--data_path', type=str, default='data/processed/tongue_fld_5folds.csv', help="Đường dẫn file dữ liệu 5-fold")
    parser.add_argument('--images_dir', type=str, default='data/raw/images', help="Thư mục ảnh")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Thư mục lưu trọng số model")
    parser.add_argument('--results_dir', type=str, default='results', help="Thư mục lưu kết quả đánh giá")
    
    args = parser.parse_args()
    
    # Thiết lập phần cứng (GPU nếu có, ngược lại CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"HỆ THỐNG HUẤN LUYỆN NCKH: CẤU HÌNH [{args.model.upper()}]")
    print(f"Thiết bị tính toán: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Đường dẫn dữ liệu: {args.data_path}")
    print("=" * 65)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    df = pd.read_csv(args.data_path)
    
    # Chạy 1 fold hoặc toàn bộ 5 folds
    folds_to_run = range(5) if args.fold == -1 else [args.fold]
    fold_results = []
    
    for f in folds_to_run:
        res = train_single_fold(f, df, args, device)
        res_record = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() for k, v in res.items()}
        res_record['fold'] = f
        fold_results.append(res_record)
        
    # Tổng kết kết quả nếu chạy cả 5 folds
    if len(fold_results) > 1:
        print("\n" + "="*30 + " TỔNG HỢP KẾT QUẢ 5-FOLD " + "="*30)
        accs = [r['accuracy'] for r in fold_results]
        f1s = [r['macro_f1'] for r in fold_results]
        maes = [r['mae'] for r in fold_results]
        rmses = [r['rmse'] for r in fold_results]
        rec2s = [r['recall_class_2'] for r in fold_results]
        
        summary = {
            'model': args.model,
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s)),
            'recall_class_2_mean': float(np.mean(rec2s)),
            'recall_class_2_std': float(np.std(rec2s)),
            'mae_mean': float(np.mean(maes)),
            'mae_std': float(np.std(maes)),
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std': float(np.std(rmses)),
            'folds_detail': fold_results
        }
        
        print(f"Mô hình: {args.model.upper()}")
        print(f"- Accuracy:   {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_std']*100:.2f}%")
        print(f"- Macro-F1:   {summary['macro_f1_mean']*100:.2f}% ± {summary['macro_f1_std']*100:.2f}%")
        print(f"- Recall Nặng: {summary['recall_class_2_mean']*100:.2f}% ± {summary['recall_class_2_std']*100:.2f}%")
        print(f"- MAE:        {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
        print(f"- RMSE:       {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
        
        # Lưu kết quả ra file JSON
        out_json = os.path.join(args.results_dir, f"{args.model}_5folds_summary.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"--> Đã lưu tóm tắt kết quả 5-Fold tại: {out_json}")


if __name__ == '__main__':
    main()
