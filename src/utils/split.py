import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(csv_path, output_path, label_col='label', n_splits=5, seed=42):
    """
    Hàm đọc file dữ liệu gốc, chia 5-fold phân tầng và lưu lại thành file cố định.
    """
    df = pd.read_csv(csv_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    df['fold'] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df[label_col])):
        df.loc[val_idx, 'fold'] = fold_idx
        
    print(f"--> Đã chia thành {n_splits} folds phân tầng thành công!")
    print("\nPhân bố các nhãn trên từng fold:")
    print(pd.crosstab(df['fold'], df[label_col]))
    
    df.to_csv(output_path, index=False)
    print(f"\n--> Đã lưu file chia fold tại: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chia 5-fold phân tầng cho bộ dữ liệu NCKH")
    parser.add_argument("--input", type=str, default="data/raw/clinical_data.csv", help="Đường dẫn file dữ liệu csv gốc")
    parser.add_argument("--output", type=str, default="data/processed/tongue_fld_5folds.csv", help="Đường dẫn lưu file 5-fold")
    parser.add_argument("--label_col", type=str, default="label", help="Tên cột nhãn mức độ bệnh")
    args = parser.parse_args()
    
    create_stratified_folds(args.input, args.output, label_col=args.label_col)
