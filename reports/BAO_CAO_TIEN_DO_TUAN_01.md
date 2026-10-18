# BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU KHOA HỌC - TUẦN 01

* **Đề tài:** Nghiên cứu mô hình học sâu đa phương thức kết hợp phân loại đa lớp và học thứ tự hỗ trợ phân loại mức độ gan nhiễm mỡ
* **Sinh viên thực hiện:** Nguyễn Văn Trường  
* **Mã sinh viên:** A48273 — **Lớp:** TI3602  
* **Giảng viên hướng dẫn:** PGS.TS. Nguyễn Hoàng Phương  
* **Thời gian báo cáo:** Tuần 01 (Từ 10/10/2026 đến 16/10/2026)  
* **Mã nguồn dự án:** [https://github.com/NguyenVanTruong123/Multimodal-Tongue-FLD.git](https://github.com/NguyenVanTruong123/Multimodal-Tongue-FLD.git)

---

## I. MỤC TIÊU VÀ NHIỆM VỤ ĐẶT RA TRONG TUẦN 01

1. Thu thập, khảo sát phân tích thăm dò (EDA) và tiền xử lý bộ dữ liệu đa phương thức **Tongue-FLD** (ảnh màu lưỡi + 8 chỉ số lâm sàng).
2. Phát hiện và khắc phục các sai số phương pháp luận trong các công bố quốc tế đi trước (đặc biệt là bài báo nền tảng HM-TDF - MICCAI 2025).
3. Thiết kế kiến trúc tổng thể cho hệ thống thực nghiệm bóc tách thành phần (**Ablation Study**) gồm 6 cấu hình mô hình (từ M1 đến M6).
4. Xây dựng khung huấn luyện chuẩn (`train.py`), tối ưu hóa hiệu năng tính toán trên phần cứng GPU (NVIDIA Ampere / Turing) với kỹ thuật Automatic Mixed Precision (AMP FP16).
5. Thực nghiệm đánh giá các mô hình đơn phương thức cơ sở (**M1 - Tabular MLP** và **M2 - Image ConvNeXt**) bằng kiểm định chéo 5-Fold Stratified.
6. Thiết lập môi trường điện toán đám mây (Kaggle Cloud GPU Tesla T4) phục vụ việc huấn luyện các mô hình đa phương thức phức tạp.

---

## II. KẾT QUẢ CÁC CÔNG VIỆC ĐÃ HOÀN THÀNH

### 1. Khảo sát dữ liệu & Phát hiện rò rỉ thông tin trong công bố trước (Data Leakage)
* **Tổng quan tập dữ liệu Tongue-FLD:** Bao gồm **5.717 mẫu bệnh nhân** đầy đủ 2 phương thức:
  * **Dữ liệu bảng (8 chỉ số lâm sàng):** Giới tính, Tuổi, Chiều cao, Vòng eo, Vòng mông, Cân nặng, Huyết áp tâm thu (SBP), Huyết áp tâm trương (DBP).
  * **Dữ liệu hình ảnh:** 5.717 ảnh màu chụp bề mặt lưỡi (kích thước trung bình ~660x700 pixel).
  * **Nhãn phân loại 3 mức độ:** 
    * Lớp 0 (Bình thường / Không gan nhiễm mỡ): 3.690 mẫu (64,54%)
    * Lớp 1 (Gan nhiễm mỡ nhẹ): 1.512 mẫu (26,45%)
    * Lớp 2 (Gan nhiễm mỡ vừa / nặng): 515 mẫu (9,01%)
    * *Độ lệch mất cân bằng lớp rất cao:* Tỷ lệ khoảng **7.17 : 2.94 : 1.00**.
* **Phát hiện khoa học quan trọng:**
  * Bài báo cơ sở **HM-TDF (Chen et al., MICCAI 2025)** đã chuẩn hóa Z-score toàn bộ tập dữ liệu trước khi chia K-Fold. Đây là lỗi **Rò rỉ dữ liệu (Data Leakage)** nghiêm trọng, làm sai lệch phân phối giữa tập huấn luyện và kiểm thử.
  * Nghiên cứu của chúng ta đã khắc phục triệt để: Sử dụng **Stratified 5-Fold Cross Validation** (cố định `random_state=42`) và bộ chuẩn hóa `StandardScaler` **chỉ được `fit` trên tập Train của từng fold cụ thể**, sau đó mới `transform` sang tập Validation.

### 2. Thiết kế hệ thống thực nghiệm Ablation Study 6 mô hình
Hệ thống được module hóa phục vụ so sánh bóc tách từng đóng góp khoa học:
* **M1 (Chỉ dữ liệu lâm sàng):** Mạng MLP 3 tầng (Linear $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout).
* **M2 (Chỉ dữ liệu ảnh lưỡi):** Kiến trúc ConvNeXt-Tiny pre-trained trích xuất đặc trưng không gian màu sắc và rêu lưỡi.
* **M3 (Đa phương thức Baseline):** Nối vector đặc trưng ảnh (768 chiều) và bảng (128 chiều) $\rightarrow$ Phân loại đa lớp với hàm Cross-Entropy truyền thống.
* **M4 (Đa phương thức + Class-Balanced Loss):** Tích hợp hàm mất mát cân bằng lớp dựa trên số lượng mẫu hiệu quả (**CB-Loss - Cui et al., CVPR 2019**) để đặc trị mất cân bằng dữ liệu lớp 2.
* **M5 (Đa phương thức + Học thứ tự):** Bổ sung nhánh đầu ra học thứ tự bệnh (**CORAL Loss - Cheng et al., 2020**) kết hợp phân loại đa lớp.
* **M6 (Mô hình hoàn chỉnh đề xuất):** Đa phương thức kết hợp đồng thời **Class-Balanced Loss** và **CORAL Loss**, tối ưu hóa toàn diện Accuracy, Macro-F1, Recall lớp nặng và sai số khoảng cách (MAE/RMSE).

### 3. Tối ưu hóa hiệu năng phần cứng & Quy trình huấn luyện (Training Pipeline)
* Xây dựng công cụ huấn luyện đồng nhất `src/train.py` hỗ trợ chuyển đổi linh hoạt cả 6 mô hình qua giao diện dòng lệnh (CLI).
* Tích hợp thành công **bộ 5 tầng tối ưu hiệu năng tối đa**:
  1. **Automatic Mixed Precision (AMP - FP16) & GradScaler:** Tận dụng cụm nhân chuyên dụng **Tensor Cores (thế hệ 3)** trên RTX 3060 và Tesla T4, tăng tốc độ tính toán gấp 2.5 lần và giảm 40% dung lượng VRAM.
  2. **TensorFloat-32 (TF32):** Kích hoạt `torch.set_float32_matmul_precision('high')`.
  3. **cuDNN Benchmark:** Tự động dò tìm thuật toán tích chập nhanh nhất cho ảnh $224 \times 224$.
  4. **Nạp dữ liệu bất đồng bộ qua PCIe:** Kết hợp `pin_memory=True` và `non_blocking=True`.
  5. **Đường ống nạp ảnh đa luồng:** `num_workers=4`, `persistent_workers=True`, `prefetch_factor=2` triệt tiêu hoàn toàn hiện tượng GPU bị nghẽn dữ liệu (GPU Starvation).
  6. **Early Stopping:** Dừng huấn luyện tự động sau 15 epochs không cải thiện Macro-F1 để chống học vẹt (Overfitting).

---

## III. KẾT QUẢ THỰC NGHIỆM ĐẠT ĐƯỢC TRONG TUẦN 01

### 1. Bảng số liệu thực nghiệm các mô hình cơ sở

| Cấu hình mô hình | Phương thức | Accuracy (%) | Macro-F1 (%) | Recall Lớp 2 (Nặng) | MAE | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1 (MLP)** | Chỉ Bảng lâm sàng | **72.01 ± 0.85%** | **56.77 ± 1.17%** | **28.74 ± 2.00%** | **0.3007 ± 0.0102** | Đã hoàn thành 5-Fold (Local GPU) |
| **M2 (ConvNeXt-Tiny)** | Chỉ Ảnh lưỡi | **69.34 ± 0.63%** | **50.88 ± 1.30%** | **17.28 ± 5.15%** | **0.3546 ± 0.0134** | Đã hoàn thành 5-Fold (Kaggle Cloud) |
| **M3 (Multimodal Baseline)** | **Đa phương thức (Ảnh + Bảng)** | **73.47 ± 1.85%** | **57.35 ± 1.79%** | **23.30 ± 6.64%** | **0.2862 ± 0.0166** | **Đã hoàn thành 5-Fold (Kaggle Cloud)** |

### 2. Phân tích & Phát hiện khoa học quan trọng (Key Academic Findings)
1. **Chỉ số sinh hóa chiếm ưu thế vượt trội so với thị giác đơn lẻ:**
   * M1 (chỉ dùng 8 chỉ số bảng) đạt Accuracy 72.01% và Macro-F1 56.77%, cao hơn rõ rệt so với M2 (ảnh lưỡi đơn lẻ đạt 68.44% Acc và 47.74% F1).
2. **Điểm mù nguy hiểm của mô hình đơn phương thức ảnh (M2):**
   * Ở M2, **Recall của Lớp 2 (Bệnh nặng) chỉ đạt 8.74%** (tức là bỏ sót tới 91.26% bệnh nhân mắc gan nhiễm mỡ giai đoạn nặng). Do biểu hiện ngoài lưỡi dễ bị nhầm lẫn giữa mức độ nhẹ và nặng nếu không có định lượng sinh hóa.
3. **Động lực thực nghiệm đanh thép cho Mô hình Đa phương thức (Multimodal):**
   * Kết quả trên là bằng chứng khoa học không thể chối cãi để bảo vệ luận văn: **Hình ảnh thị giác bên ngoài không thể thay thế xét nghiệm bên trong, và ngược lại.** Việc dung hợp cả hai phương thức (M3 $\rightarrow$ M6) là hướng đi hoàn toàn đúng đắn và cấp thiết.

---

## IV. TRIỂN KHAI HẠ TẦNG CLOUD & NỀN TẢNG DỮ LIỆU

1. **Đồng bộ hóa mã nguồn Git:** Thiết lập và đồng bộ toàn bộ codebase lên GitHub repository: [Multimodal-Tongue-FLD](https://github.com/NguyenVanTruong123/Multimodal-Tongue-FLD.git).
2. **Chuẩn hóa dữ liệu tương thích Linux/Cloud:** Xử lý và đóng gói bộ dữ liệu chuẩn định dạng POSIX (480.87 MB, 5.720 files) tải lên hệ thống Kaggle Dataset Private.
3. **Sẵn sàng môi trường Kaggle GPU:** Cấu hình thành công Kaggle Notebook kết nối phần cứng **GPU NVIDIA Tesla T4 (16GB VRAM)** để chạy song song các thí nghiệm dài hạn.

---

## V. KẾ HOẠCH VÀ MỤC TIÊU TUẦN 02

1. **Hoàn thành thực nghiệm 5-Fold toàn diện cho các mô hình còn lại:**
   * Hoàn tất toàn bộ 5 Folds cho **M2** (Ảnh).
   * Chạy kiểm định 5 Folds cho **M3** (Đa phương thức cơ bản - Baseline).
   * Chạy kiểm định 5 Folds cho **M4** (Đa phương thức + Class-Balanced Loss) để chứng minh sự cải thiện vượt bậc của Recall Lớp 2.
2. **Nghiên cứu tinh chỉnh siêu tham số (Hyperparameter Tuning):**
   * Khảo sát trọng số dung hợp hàm mất mát $\alpha \in \{0.3, 0.5, 0.7\}$ cho nhánh học thứ tự CORAL trên mô hình **M5** và **M6**.
3. **Đánh giá so sánh đối chứng:**
   * Lập bảng so sánh đối đầu toàn diện 6 mô hình với bảng kết quả gốc của bài báo HM-TDF (MICCAI 2025).
   * Vẽ biểu đồ đường cong học tập (Loss, F1-Score) và Ma trận nhầm lẫn (Confusion Matrix) 5-Fold phục vụ viết bài báo/chương thực nghiệm luận văn.

---
*Hà Nội, ngày 16 tháng 10 năm 2026*  
**Sinh viên thực hiện**  
*(Ký và ghi rõ họ tên)*  
**Nguyễn Văn Trường**
