# Báo Cáo Thực Hành LAB 16: Cloud AI Environment Setup (GCP)

- **Họ và tên:** Lê Quang Thành
- **Email:** quangthanh17022004@gmail.com
- **GCP Project ID:** `ai-lab-16-gcp-510516`
- **Region / Zone:** `us-central1` / `us-central1-a`
- **Cấu hình máy ảo:** `e2-medium` (2 vCPU, 4GB RAM, 30GB SSD, Debian 12 Linux 6.1 x86_64)
- **Ngày thực hiện:** 04/10/2026

---

## 1. Bảng Kết quả Benchmark LightGBM trên CPU Node (`e2-medium`)

*(Dữ liệu đo đạc thực tế từ quá trình chạy `python3 benchmark.py` trên Compute Node GCP)*

| Metric | Kết quả thực tế | Đơn vị / Ghi chú |
|---|---|---|
| **Thời gian load data** | **2.953** | giây (s) cho 284,807 dòng CSV (144MB) |
| **Thời gian training** | **1.978** | giây (s) cho 205,060 dòng train |
| **Best iteration** | **1** | vòng lặp tối ưu |
| **AUC-ROC** | **0.9267** | đo theo xác suất `predict_proba` |
| **Accuracy** | **0.9989 (99.89%)** | tỷ lệ dự đoán chính xác toàn cục |
| **F1-Score** | **0.7382** | độ cân bằng giữa Precision và Recall |
| **Precision** | **0.6370** | tỷ lệ dự báo gian lận thực sự đúng |
| **Recall** | **0.8776** | tỷ lệ phát hiện các ca gian lận thực tế |
| **Inference latency (1 row)** | **1.391** | ms / sample (trung bình 100 lần lặp) |
| **Inference throughput (1000 rows)** | **559,602.9** | samples / giây (đo theo lô 1,000 dòng) |

---

## 2. Nhận xét & Đánh giá Kết quả (Báo cáo phân tích)

1. **Về thời gian nạp và huấn luyện dữ liệu trên CPU:**
   - Bộ dữ liệu `creditcard.csv` có dung lượng 144MB với 284,807 dòng được nạp hoàn tất chỉ trong **2.953s**.
   - Thời gian huấn luyện LightGBM trên máy ảo CPU nhỏ `e2-medium` (2 vCPU) diễn ra cực nhanh, chỉ mất **1.978 giây** cho hơn 205,000 mẫu. Điều này chứng minh hiệu quả vượt trội của thuật toán Histogram-based Decision Tree của LightGBM, hoàn toàn có thể train nhanh gọn trên CPU giá rẻ (~$0.033/h) mà không bắt buộc phải dùng tới GPU tốn kém.

2. **Về chất lượng dự báo (Metrics):**
   - Chỉ số **AUC-ROC đạt 0.9267** và **Recall đạt 87.76%**, cho thấy mô hình có khả năng bắt được phần lớn các giao dịch gian lận trong thực tế — đây là mục tiêu sống còn trong bài toán tài chính/ngân hàng.
   - Do tỷ lệ gian lận trong tập dữ liệu cực thấp (chỉ ~0.17%), chỉ số **Accuracy đạt 99.89%** nhưng không phản ánh toàn diện, do đó việc kết hợp **F1-Score (0.7382)** và **AUC-ROC** là bắt buộc để đánh giá đúng năng lực phân loại.

3. **Về hiệu năng suy luận (Inference):**
   - **Độ trễ suy luận 1 dòng (Latency):** Chỉ **1.391 ms/giao dịch**, hoàn toàn thỏa mãn yêu cầu của các hệ thống chấm điểm rủi ro thời gian thực (Real-time Transaction Fraud Detection với SLA < 50ms).
   - **Thông lượng (Throughput):** Đạt mức ấn tượng **559,602 dòng/giây**, cho phép hệ thống phân tích xử lý hàng triệu giao dịch theo lô trong thời gian ngắn mà không làm nghẽn tài nguyên CPU.

---

## 3. Quan sát và Đánh giá Tài nguyên (CP4)

*(Ghi nhận thực tế sau khi hoàn thành huấn luyện và đo đạc)*

### 3.1. Thống kê sử dụng tài nguyên (CPU, RAM, Network)
- **CPU (Lệnh `top`):**
  - Trạng thái Idle: `%Cpu(s): 0.0 us, 0.0 sy, 100.0 id` (hệ thống giải phóng 100% CPU ngay sau khi train xong).
  - Load average: `0.00, 0.04, 0.06` (mức tải hệ thống cực kỳ an toàn, không có hiện tượng nghẽn I/O hay CPU throttle).
- **RAM (Lệnh `free -h`):**
  - Tổng RAM: **3.8 GiB** (cấu hình `e2-medium`).
  - RAM sử dụng (Used): **507 MiB** (~13% tổng RAM).
  - Buffer / Cache: **986 MiB** (hệ điều hành giữ bộ đệm file dữ liệu để tối ưu đọc ghi).
  - RAM khả dụng (Available): **3.3 GiB** (hơn 85% dung lượng còn trống, đảm bảo an toàn tuyệt đối, không kích hoạt Swap).
- **Network Traffic (Lệnh `ip -s link` trên interface `ens4`):**
  - **RX (Nhận):** **255,194,950 bytes (~255.2 MB)** và 18,554 packets. Lưu lượng này phản ánh chính xác quá trình tải các thư viện Python (`pip install`) và tải bộ dữ liệu `creditcardfraud` từ Kaggle qua Cloud NAT Gateway.
  - **TX (Gửi):** **1,148,648 bytes (~1.15 MB)** và 9,636 packets (chủ yếu là traffic SSH IAP và các request API outgoing).
  - **Errors / Dropped:** 0 gói tin bị rớt, kết nối mạng VPC và Cloud NAT hoạt động ổn định 100%.

---

## 4. Phân tích Chi phí Thực tế & Ước tính Billing (CP4)

### 4.1. Bảng ước tính chi phí hạ tầng (Region `us-central1`)
| Dịch vụ (Service) | Loại tài nguyên | Chi phí / giờ | Chi phí thực tế buổi lab (~30 phút) |
|---|---|---|---|
| **Compute Engine** | Máy ảo `e2-medium` (2 vCPU, 4GB RAM) | ~$0.033 / giờ | ~$0.0165 |
| **Cloud NAT** | Gateway + Egress traffic (~0.26 GB) | ~$0.044 / giờ + data | ~$0.0225 |
| **Cloud Load Balancing** | External HTTP LB Forwarding Rule | ~$0.008 / giờ | ~$0.0040 |
| **Persistent Disk** | 30 GB Standard SSD Boot Disk | ~$0.002 / giờ | ~$0.0010 |
| **TỔNG CỘNG** | | **~$0.087 / giờ** | **~$0.044 (~1.150 VNĐ)** |

### 4.2. Ghi chú về trạng thái GCP Billing
- **Thời điểm quan sát:** 04/10/2026.
- **Tình trạng:** Google Cloud Billing Dashboard hiển thị **$0.00** do hệ thống ghi nhận cước của Google Cloud có **độ trễ từ 12 đến 24 giờ** để xử lý hóa đơn chi tiết theo SKU.
- Toàn bộ chi phí thực tế (~$0.044) hoàn toàn nằm trong gói tài khoản **Google Cloud Free Trial ($300 credits)** và không phát sinh bất kỳ khoản phí ngoài ý muốn nào.

---

## 5. Trả lời các câu hỏi kỹ thuật trong Lab

1. **Vì sao AUC-ROC phải dùng xác suất (`predict_proba`) thay vì nhãn cứng (0 hoặc 1)?**
   - *Trả lời:* Đường cong ROC được xây dựng bằng cách tính toán True Positive Rate (TPR) và False Positive Rate (FPR) trên **tất cả các ngưỡng quyết định (threshold) từ 0 đến 1**. Nếu chỉ sử dụng nhãn nhị phân cứng (0 hoặc 1), mô hình sẽ chỉ tạo ra một điểm duy nhất trên không gian ROC, làm mất hoàn toàn khả năng tinh chỉnh ngưỡng cắt (threshold tuning) cho các bài toán dữ liệu mất cân bằng như phát hiện gian lận.

2. **Vì sao Throughput có đơn vị dòng/giây và thời gian inference bao gồm những gì?**
   - *Trả lời:* Throughput đo lường năng lực phục vụ đồng thời của hệ thống trong một đơn vị thời gian (1 giây), tính bằng công thức: `1000 / thời_gian_dự_đoán_1000_dòng (giây)`. Thời gian inference **chỉ tính riêng** thời gian CPU thực hiện việc duyệt cây quyết định để tính toán kết quả đầu ra cho các mẫu dữ liệu, **không bao gồm** thời gian khởi tạo VM, nạp dữ liệu hay huấn luyện mô hình.

---

## 6. Checklist Minh chứng Nộp bài (Deliverables)

- [x] **1. Screenshot terminal chạy `python3 benchmark.py`** hiển thị toàn bộ output kết quả và JSON.
- [x] **2. File `benchmark_result.json`** đã lưu trên máy cá nhân (`./benchmark_result.json`).
- [x] **3. Screenshot tài nguyên VM:** lệnh `top`, `free -h`, `ip -s link` trên terminal (kèm số liệu chi tiết ở Mục 3).
- [ ] **4. Screenshot GCP Billing Reports:** trang Billing -> Reports thể hiện bộ lọc Project `ai-lab-16-gcp-510516` (kèm ghi chú độ trễ ở Mục 4.2).
- [x] **5. File nén mã nguồn:** file `terraform-gcp.zip`.
- [x] **6. Báo cáo hoàn chỉnh:** file `REPORT_TEMPLATE.md` (đã điền đầy đủ 100% nội dung).
- [ ] **7. Dọn dẹp tài nguyên:** Chạy `terraform destroy` để hủy tài nguyên GCP.
