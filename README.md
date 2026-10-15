# Nghiên cứu mô hình học sâu đa phương thức hỗ trợ phân loại mức độ gan nhiễm mỡ

- **Chủ nhiệm đề tài:** Nguyễn Văn Trường (MSV: A48273 - Lớp: TI3602)
- **Cán bộ hướng dẫn:** PGS.TS. Nguyễn Hoàng Phương
- **Đơn vị:** Khoa Công Nghệ Thông Tin – Trường Đại học Thăng Long
- **Bộ dữ liệu thực nghiệm:** Tongue-FLD (5.717 mẫu)

---

## 1. Cấu trúc thư mục dự án

```text
NCKH/
├── data/
│   ├── raw/                        # Chứa ảnh gốc và file chỉ số sinh lý ban đầu
│   │   ├── images/                 # 5.717 ảnh lưỡi đã phân đoạn
│   │   └── clinical_data.csv       # File bảng chứa 8 chỉ số sinh lý
│   └── processed/
│       └── tongue_fld_5folds.csv   # File sau khi chia 5-fold phân tầng cố định
├── notebooks/                      # Thử nghiệm EDA, trực quan hóa dữ liệu
├── src/
│   ├── models/                     # Kiến trúc mô hình (M1, M2, Multimodal)
│   ├── losses/                     # Hàm Class-Balanced Loss và CORAL Loss
│   ├── utils/                      # Độ đo (metrics.py) và chia dữ liệu (split.py)
│   └── dataset.py                  # PyTorch Dataset hỗ trợ nạp dữ liệu
├── checkpoints/                    # Nơi lưu file trọng số mô hình tốt nhất (.pth)
├── results/                        # Nơi lưu kết quả đánh giá, ma trận nhầm lẫn
├── requirements.txt                # Danh sách thư viện cần thiết
└── README.md
```

---

## 2. Hệ thống thí nghiệm đối chứng (Ablation Study)

Đề tài so sánh 6 cấu hình thực nghiệm trên cùng bộ chia 5-fold:

1. **M1:** Chỉ sử dụng 8 chỉ số sinh lý (Mạng MLP).
2. **M2:** Chỉ sử dụng ảnh lưỡi (ConvNeXt-Tiny).
3. **M3:** Kết hợp ảnh lưỡi + chỉ số sinh lý với Cross-Entropy thông thường (Baseline Multimodal).
4. **M4:** M3 + Class-Balanced Cross-Entropy (xử lý mất cân bằng mẫu).
5. **M5:** Mô hình hai đầu ra: Cross-Entropy + CORAL Loss (học thứ tự bệnh).
6. **M6 (Đề xuất):** Mô hình hai đầu ra: **Class-Balanced Loss + CORAL Loss**.

---

## 3. Các bước tiếp theo

1. Đặt dữ liệu Tongue-FLD vào thư mục `data/raw/`.
2. Chạy chia 5-fold phân tầng:
   ```bash
   python src/utils/split.py --input data/raw/clinical_data.csv --output data/processed/tongue_fld_5folds.csv
   ```
3. Bắt đầu huấn luyện mô hình M1 trước để kiểm tra pipeline.
