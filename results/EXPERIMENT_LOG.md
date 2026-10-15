# SỔ TAY NHẬT KÝ THỰC NGHIỆM KHOA HỌC (EXPERIMENT LOG)

* **Tên đề tài:** Nghiên cứu mô hình học sâu đa phương thức kết hợp phân loại đa lớp và học thứ tự hỗ trợ phân loại mức độ gan nhiễm mỡ
* **Chủ nhiệm đề tài:** Nguyễn Văn Trường (MSV: A48273 - Lớp: TI3602)
* **Cán bộ hướng dẫn:** PGS.TS. Nguyễn Hoàng Phương
* **Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Thăng Long
* **Bộ dữ liệu thực nghiệm:** Tongue-FLD (5.717 mẫu: 3.690 Không mắc, 1.512 Nhẹ, 515 Vừa/Nặng)
* **Kho lưu trữ mã nguồn:** [GitHub: Multimodal-Tongue-FLD](https://github.com/NguyenVanTruong123/Multimodal-Tongue-FLD)

---

## 📌 BẢNG THEO DÕI TIẾN ĐỘ THỰC NGHIỆM ĐỐI CHỨNG (ABLATION STUDY)

Hệ thống đánh giá 6 cấu hình mô hình trên cùng bộ chia Stratified 5-Fold cố định:

| Cấu hình | Dữ liệu đầu vào | Hàm mất mát (Loss) | Accuracy (%) | Macro-F1 (%) | Recall Nặng (%) | MAE (↓) | Trạng thái |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M1** | 8 chỉ số sinh lý (MLP) | Cross-Entropy | **72.01 ± 0.85** | **56.77 ± 1.17** | **28.74 ± 2.00** | **0.3007 ± 0.0102** | ✅ **Hoàn thành** |
| **M2** | Ảnh lưỡi (ConvNeXt-Tiny) | Cross-Entropy | *Đang chờ* | *Đang chờ* | *Đang chờ* | *Đang chờ* | ⏳ Sắp tới |
| **M3** | Đa phương thức (Ảnh + Bảng) | Cross-Entropy | *Đang chờ* | *Đang chờ* | *Đang chờ* | *Đang chờ* | ⏳ Sắp tới |
| **M4** | M3 + Cân bằng lớp | Class-Balanced CE | *Đang chờ* | *Đang chờ* | *Đang chờ* | *Đang chờ* | ⏳ Sắp tới |
| **M5** | M3 + Học thứ tự (Dual-Head) | CE + CORAL Loss | *Đang chờ* | *Đang chờ* | *Đang chờ* | *Đang chờ* | ⏳ Sắp tới |
| **M6** | **Mô hình đề xuất hoàn chỉnh** | **CB-Loss + CORAL Loss** | *Đang chờ* | *Đang chờ* | *Đang chờ* | *Đang chờ* | ⏳ Sắp tới |

---

## 🔬 GIAI ĐOẠN 1: KHẢO SÁT, TIỀN XỬ LÝ DỮ LIỆU & PHÂN CHIA 5-FOLD
* **Ngày thực hiện:** 15/10/2026
* **Môi trường & Notebook:** `notebooks/01_eda_and_preprocessing.ipynb`
* **File kết quả sinh ra:** `data/processed/tongue_fld_5folds.csv` (Commit: `97b2897`)

### 1. Kết quả khảo sát đặc tính bộ dữ liệu Tongue-FLD:
* **Quy mô:** $5.717$ mẫu đầy đủ, tỷ lệ missing data là $0\%$.
* **Phân bố nhãn bệnh (Class Distribution) mất cân bằng nghiêm trọng:**
  * Lớp 0 (Không mắc - Non-FLD): **3.690 mẫu** (tỷ lệ **64.54%**)
  * Lớp 1 (Mức độ nhẹ - Mild FLD): **1.512 mẫu** (tỷ lệ **26.45%**)
  * Lớp 2 (Mức độ vừa / nặng - Moderate/Severe FLD): **515 mẫu** (tỷ lệ **9.01%**)
  * Tỷ số mất cân bằng: **7.17 : 2.94 : 1.00**.
* **Đặc tính chỉ số lâm sàng:** Vòng eo (`Waist`), Tỷ số eo/hông (`Waist`/`Hipline`), Cân nặng (`Weight`) và Huyết áp (`SBP`) có tương quan thuận rõ rệt với mức độ tiến triển của bệnh gan nhiễm mỡ.

### 2. Phát hiện khoa học quan trọng về lỗi của bài báo gốc (HM-TDF):
* **Lỗi rò rỉ dữ liệu (Data Leakage):** Nhóm tác giả bài báo HM-TDF (MICCAI 2025) đã chuẩn hóa Z-score toàn bộ dữ liệu trước khi chia K-Fold (dòng 463-464 file `main.py`). Điều này làm cho thông tin phân phối thống kê (mean, std) của tập Validation bị rò rỉ vào tập Train, khiến kết quả đánh giá có nguy cơ bị thổi phồng ảo.
* **Giải pháp khắc phục của đề tài:** 
  1. Phân chia dữ liệu trước bằng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` cố định vào file `tongue_fld_5folds.csv`.
  2. Áp dụng nguyên tắc chuẩn hóa chuẩn khoa học: **Chỉ fit StandardScaler trên tập Train của từng fold, sau đó dùng tham số đó để transform sang tập Validation**. Đảm bảo tính khách quan tuyệt đối.

---

## 🚀 GIAI ĐOẠN 2: THỰC NGHIỆM MÔ HÌNH M1 (TABULAR MLP BASELINE)
* **Ngày thực hiện:** 15/10/2026
* **Tập lệnh thực thi:** `python src/train.py --model m1 --fold -1 --epochs 50`
* **File lưu kết quả:** `results/m1_5folds_summary.json`
* **File lưu trọng số tốt nhất:** `checkpoints/best_m1_fold_0.pth` $ightarrow$ `best_m1_fold_4.pth`

### 1. Mục tiêu thí nghiệm M1:
* Thiết lập kết quả mốc cơ sở (Baseline benchmark) khi **chỉ sử dụng 8 chỉ số sinh lý lâm sàng** đo bằng phương pháp truyền thống (chưa tích hợp ảnh lưỡi).
* Kiểm tra toàn diện sự thông suốt của pipeline huấn luyện: Data Loading $ightarrow$ Chuẩn hóa không rò rỉ $ightarrow$ Huấn luyện MLP $ightarrow$ Early Stopping $ightarrow$ Đánh giá đa độ đo.

### 2. Cấu hình kiến trúc & Siêu tham số M1:
* **Kiến trúc mạng:** Multi-Layer Perceptron (MLP) 3 tầng:
  * Tầng vào: 8 nơ-ron (Gender, Age, Height, Waist, Hipline, Weight, SBP, DBP)
  * Tầng ẩn 1: `Linear(8, 64)` $ightarrow$ `BatchNorm1d(64)` $ightarrow$ `ReLU()` $ightarrow$ `Dropout(0.3)`
  * Tầng ẩn 2: `Linear(64, 32)` $ightarrow$ `BatchNorm1d(32)` $ightarrow$ `ReLU()` $ightarrow$ `Dropout(0.3)`
  * Tầng ra: `Linear(32, 3)` (3 mức độ phân loại)
* **Thuật toán tối ưu:** `AdamW` (Learning rate: $10^{-3}$, Weight decay: $10^{-4}$)
* **Bộ điều chỉnh tốc độ học:** `CosineAnnealingLR` ($T_{max} = 50$, $\eta_{min} = 10^{-6}$)
* **Kích thước Batch:** 64
* **Cơ chế dừng sớm (Early Stopping):** `patience = 15` epochs theo dõi `Macro-F1`.

### 3. Bảng số liệu chi tiết qua 5 Folds:

| Fold kiểm định | Epoch tối ưu | Train Loss | Val Loss | Accuracy (%) | Macro-F1 (%) | Recall Lớp 0 (%) | Recall Lớp 1 (%) | Recall Lớp 2 (%) | MAE (↓) | RMSE (↓) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 18 | 0.6440 | 0.6400 | 71.50 | 56.79 | 88.21 | 44.88 | 30.10 | 0.3077 | 0.5943 |
| **Fold 1** | 7 | 0.6559 | 0.6428 | 70.72 | 54.79 | 89.02 | 39.60 | 31.07 | 0.3173 | 0.6052 |
| **Fold 2** | 15 | 0.6518 | 0.6285 | 73.23 | 57.48 | 89.97 | 47.68 | 28.16 | 0.2922 | 0.5841 |
| **Fold 3** | 32 | 0.6334 | 0.6259 | 72.35 | 58.28 | 87.13 | 50.99 | 29.13 | 0.2913 | 0.5666 |
| **Fold 4** | 44 | 0.6329 | 0.6252 | 72.27 | 56.51 | 87.94 | 50.00 | 25.24 | 0.2948 | 0.5743 |
| **TRUNG BÌNH** | — | — | — | **72.01 ± 0.85** | **56.77 ± 1.17** | **88.45 ± 1.05** | **46.63 ± 4.19** | **28.74 ± 2.00** | **0.3007 ± 0.0102** | **0.5849 ± 0.0138** |

---

### 4. Phân tích chuyên sâu & Nhận xét khoa học về Mô hình M1:

#### a. Điểm mạnh và tính ổn định của mô hình:
1. **Độ ổn định thực nghiệm rất cao:** Độ lệch chuẩn của Accuracy đạt mức cực kỳ ấn tượng ($\pm 0.85\%$), Macro-F1 ($\pm 1.17\%$). Điều này chứng minh quy trình Stratified 5-Fold hoạt động chuẩn xác, phân phối dữ liệu đồng đều qua các fold và kết quả không bị phụ thuộc vào may rủi ngẫu nhiên.
2. **Khả năng tổng quát hóa tốt, không bị Overfitting:**
   * Tại các epoch tối ưu, khoảng cách giữa `Train Loss` ($pprox 0.63 - 0.65$) và `Val Loss` ($pprox 0.62 - 0.64$) là cực kỳ nhỏ.
   * Kỹ thuật điều hòa (BatchNorm + Dropout 0.3 + Weight Decay) đã kiểm soát hoàn hảo hiện tượng học vẹt trên dữ liệu bảng.
3. **Giá trị chẩn đoán độc lập của các chỉ số sinh lý:**
   * Chỉ với 8 chỉ số đơn giản, mô hình đạt Accuracy **72.01%** và Macro-F1 **56.77%**.
   * *So sánh đối chiếu với bài báo HM-TDF:* Mô hình ResNet34 (chỉ dùng ảnh lưỡi) trong Table 1 của bài báo đạt Macro-F1 **56.70%** và MAE **0.3097**. Mô hình M1 của chúng ta chỉ dùng bảng đã đạt Macro-F1 **56.77%** và MAE **0.3007** (tương đương và nhỉnh hơn một chút so với ResNet34 dùng ảnh). Điều này khẳng định 8 chỉ số sinh lý mang thông tin chẩn đoán bệnh nền tảng rất quan trọng.

#### b. Điểm hạn chế chí tử ("Gót chân Asin") của M1:
* **Sự thiên vị trầm trọng về lớp đa số (Majority Class Bias):**
  * Mô hình nhận diện lớp 0 (Không mắc bệnh) cực kỳ xuất sắc với **Recall = 88.45%**.
  * Nhưng với lớp 2 (Mức độ vừa / nặng), **Recall chỉ đạt vỏn vẹn 28.74%**!
  * Có tới **hơn 71% bệnh nhân gan nhiễm mỡ nặng bị mô hình bỏ sót hoặc chẩn đoán nhầm sang mức độ nhẹ**.
* **Nguyên nhân cốt lõi:**
  1. Do hàm mất mát Cross-Entropy tiêu chuẩn đối xử bình đẳng với mọi mẫu, dẫn tới việc mô hình tập trung tối ưu cho lớp chiếm 64.5% số lượng mà bỏ qua lớp thiểu số chỉ chiếm 9.0%.
  2. Dữ liệu bảng 8 chỉ số không đủ chiều sâu để phân tách ranh giới tinh vi giữa ca bệnh nhẹ và ca bệnh đã chuyển sang giai đoạn vừa/nặng.

#### c. Luận điểm khoa học bảo vệ đề tài (Đóng góp của M1 vào chuỗi nghiên cứu):
> **Kết luận học thuật:**
> *"Thực nghiệm M1 đã chứng minh rõ ràng rằng: Việc chỉ dựa vào các chỉ số sinh lý lâm sàng truyền thống là KHÔNG ĐỦ để sàng lọc bệnh nhân gan nhiễm mỡ, bởi mô hình bỏ sót tới 71.26% các ca bệnh nặng nguy hiểm (Recall chỉ đạt 28.74%). Phát hiện này là bằng chứng thực nghiệm then chốt, khẳng định tính cấp thiết của đề tài trong việc:*
> *1. Phải tích hợp thêm đặc trưng thị giác từ ảnh chụp lưỡi (M2, M3) để cung cấp thêm thông tin màu sắc, rêu lưỡi.*
> *2. Phải ứng dụng hàm mất mát Class-Balanced Loss và cơ chế học thứ tự CORAL (M4, M5, M6) để khắc phục triệt để sự thiên vị mẫu, kéo tỷ lệ phát hiện bệnh nhân nặng lên mức an toàn trong chẩn đoán y tế."*

---

## 📅 KẾ HOẠCH BƯỚC TIẾP THEO:
1. Cài đặt bản PyTorch CUDA 12.1 để kích hoạt sức mạnh card đồ họa rời **NVIDIA GeForce RTX 3060**.
2. Huấn luyện mô hình **M2 (Image Only - ConvNeXt-Tiny)** trên 5.717 ảnh lưỡi để đánh giá sức mạnh độc lập của nguồn dữ liệu hình ảnh.
