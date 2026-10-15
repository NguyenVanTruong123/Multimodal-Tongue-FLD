"""
HỆ THỐNG HUẤN LUYỆN VÀ ĐÁNH GIÁ 5-FOLD CHO BÀI TOÁN PHÂN LOẠI GAN NHIỄM MỠ (TONGUE-FLD)
Tối ưu hóa tối đa hiệu năng GPU NVIDIA RTX (Ampere) với Automatic Mixed Precision (AMP), TF32, cuDNN Benchmark và Async Data Transfer.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_absolute_error, root_mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Đảm bảo UTF-8 cho terminal Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import các mô hình và loss function
from models.m1_mlp import M1_MLP
from models.m2_convnext import M2_ConvNeXt
from models.multimodal import MultimodalFattyLiverModel
from losses.class_balanced_loss import ClassBalancedCrossEntropyLoss
from losses.coral_loss import CoralLoss


def set_seed(seed=42):
    """Cố định seed ngẫu nhiên để tái lập kết quả"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms():
    """Định nghĩa pipeline tiền xử lý và tăng cường dữ liệu ảnh"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class TongueFLDDataset(Dataset):
    """
    Dataset linh hoạt hỗ trợ cả 3 chế độ:
    - 'tabular_only': chỉ trả về (tabular, label) -> dùng cho M1
    - 'image_only': chỉ trả về (image, label) -> dùng cho M2
    - 'multimodal': trả về (image, tabular, label) -> dùng cho M3, M4, M5, M6
    """
    def __init__(self, df, image_dir, feature_cols, transform=None, mode='multimodal'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.feature_cols = feature_cols
        self.transform = transform
        self.mode = mode
        
        self.labels = self.df['label'].values.astype(np.int64)
        if self.mode in ['tabular_only', 'multimodal']:
            self.tabular_data = self.df[self.feature_cols].values.astype(np.float32)
        if self.mode in ['image_only', 'multimodal']:
            self.image_names = self.df['image_name'].values
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.mode == 'tabular_only':
            tab = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
            return tab, label
            
        # Nạp ảnh cho image_only hoặc multimodal
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'image_only':
            return image, label
            
        # multimodal
        tab = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        return image, tab, label


def prepare_fold_data(df, fold, numeric_cols, feature_cols):
    """
    Chia tách Train/Val cho fold cụ thể và chuẩn hóa CHỐNG RÒ RỈ THÔNG TIN (Anti Data Leakage):
    StandardScaler CHỈ fit trên tập Train của fold, sau đó transform sang Val.
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


def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_coral, model_name, alpha, device, scaler=None):
    """Vòng lặp huấn luyện trong 1 epoch hỗ trợ tăng tốc AMP (Automatic Mixed Precision)"""
    model.train()
    total_loss = 0.0
    use_amp = (scaler is not None and device.type == 'cuda')
    
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if model_name == 'm1':
                tabular, labels = batch
                tabular = tabular.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(tabular)
                loss = criterion_cls(logits, labels)
                
            elif model_name == 'm2':
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion_cls(logits, labels)
                
            elif model_name in ['m3', 'm4']:
                images, tabular, labels = batch
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits, _ = model(images, tabular)
                loss = criterion_cls(logits, labels)
                
            elif model_name in ['m5', 'm6']:
                images, tabular, labels = batch
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits, coral_logits = model(images, tabular)
                loss_cls = criterion_cls(logits, labels)
                loss_coral = criterion_coral(coral_logits, labels)
                loss = alpha * loss_cls + (1.0 - alpha) * loss_coral
                
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * labels.size(0)
        
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion_cls, criterion_coral, model_name, alpha, device):
    """Vòng lặp đánh giá trên tập Validation có hỗ trợ tăng tốc AMP"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    use_amp = (device.type == 'cuda')
    
    for batch in dataloader:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if model_name == 'm1':
                tabular, labels = batch
                tabular = tabular.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(tabular)
                loss = criterion_cls(logits, labels)
                preds = torch.argmax(logits, dim=1)
                
            elif model_name == 'm2':
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion_cls(logits, labels)
                preds = torch.argmax(logits, dim=1)
                
            elif model_name in ['m3', 'm4']:
                images, tabular, labels = batch
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits, _ = model(images, tabular)
                loss = criterion_cls(logits, labels)
                preds = torch.argmax(logits, dim=1)
                
            elif model_name in ['m5', 'm6']:
                images, tabular, labels = batch
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits, coral_logits = model(images, tabular)
                loss_cls = criterion_cls(logits, labels)
                loss_coral = criterion_coral(coral_logits, labels)
                loss = alpha * loss_cls + (1.0 - alpha) * loss_coral
                preds = torch.argmax(logits, dim=1)
                
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics['val_loss'] = total_loss / len(dataloader.dataset)
    return metrics


def compute_metrics(y_true, y_pred):
    """Tính toán đầy đủ bộ chỉ số đánh giá y tế đa lớp và thứ tự"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'macro_f1': f1,
        'recall_class_0': recalls[0] if len(recalls) > 0 else 0.0,
        'recall_class_1': recalls[1] if len(recalls) > 1 else 0.0,
        'recall_class_2': recalls[2] if len(recalls) > 2 else 0.0,
        'mae': mae,
        'rmse': rmse
    }


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
    
    # Cấu hình nạp dữ liệu đa luồng, pin_memory và prefetch
    pin_mem = (device.type == 'cuda')
    persistent = (args.num_workers > 0)
    prefetch = 2 if args.num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch
    )
    
    # Khởi tạo mô hình mới tinh (không nhớ kiến thức fold khác)
    model = build_model(args.model, num_classes=3, tabular_dim=len(feature_cols), dropout_rate=args.dropout).to(device)
    
    # Thiết lập hàm mất mát
    criterion_coral = CoralLoss()
    if args.model in ['m1', 'm2', 'm3', 'm5']:
        criterion_cls = nn.CrossEntropyLoss()
    elif args.model in ['m4', 'm6']:
        samples_per_cls = train_df['label'].value_counts().sort_index().values
        criterion_cls = ClassBalancedCrossEntropyLoss(samples_per_cls=samples_per_cls, beta=args.cb_beta)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_macro_f1 = -1.0
    best_metrics = None
    best_epoch = -1
    no_improve_epochs = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{args.model}_fold_{fold}.pth")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_coral, 
            args.model, args.alpha, device, scaler=scaler
        )
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
            
        # Early Stopping theo dõi Macro-F1
        if no_improve_epochs >= args.patience:
            print(f"--> Kích hoạt Early Stopping tại Epoch {epoch} (Không cải thiện Macro-F1 trong {args.patience} epochs)")
            break
            
    print(f"\n[KẾT QUẢ TỐT NHẤT FOLD {fold}] (Epoch {best_epoch}):")
    print(f"  - Accuracy:     {best_metrics['accuracy']*100:.2f}%")
    print(f"  - Macro-F1:     {best_metrics['macro_f1']*100:.2f}%")
    print(f"  - Recall Lớp 0: {best_metrics['recall_class_0']*100:.2f}%")
    print(f"  - Recall Lớp 1: {best_metrics['recall_class_1']*100:.2f}%")
    print(f"  - Recall Lớp 2: {best_metrics['recall_class_2']*100:.2f}% (Gan nhiễm mỡ nặng)")
    print(f"  - MAE:          {best_metrics['mae']:.4f}")
    print(f"  - RMSE:         {best_metrics['rmse']:.4f}")
    
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
    parser.add_argument('--num_workers', type=int, default=4, help="Số worker cho DataLoader nạp ảnh đa luồng")
    parser.add_argument('--data_path', type=str, default='data/processed/tongue_fld_5folds.csv', help="Đường dẫn file dữ liệu 5-fold")
    parser.add_argument('--images_dir', type=str, default='data/raw/images', help="Thư mục ảnh")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Thư mục lưu trọng số model")
    parser.add_argument('--results_dir', type=str, default='results', help="Thư mục lưu kết quả đánh giá")
    
    args = parser.parse_args()
    
    # Tối ưu hóa phần cứng NVIDIA Ampere (RTX 3060)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cpu')
        gpu_name = 'CPU'
        
    print("=" * 65)
    print(f"HỆ THỐNG HUẤN LUYỆN NCKH: CẤU HÌNH [{args.model.upper()}]")
    print(f"Thiết bị tính toán: {device} ({gpu_name})")
    print(f"Tối ưu hiệu năng: cuDNN Benchmark + TF32 + AMP (FP16) + Async Data Transfer")
    print(f"Đường dẫn dữ liệu: {args.data_path}")
    print("=" * 65)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    df = pd.read_csv(args.data_path)
    
    folds_to_run = range(5) if args.fold == -1 else [args.fold]
    fold_results = []
    
    for f in folds_to_run:
        res = train_single_fold(f, df, args, device)
        res_record = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() for k, v in res.items()}
        res_record['fold'] = f
        fold_results.append(res_record)
        
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
        
        out_json = os.path.join(args.results_dir, f"{args.model}_5folds_summary.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"--> Đã lưu tóm tắt kết quả 5-Fold tại: {out_json}")


if __name__ == '__main__':
    main()
