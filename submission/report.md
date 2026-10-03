# Báo Cáo Thực Hành LAB 16: Cloud AI Environment Setup (GCP)

- **Họ và tên:** Lê Quang Thành
- **Email:** quangthanh17022004@gmail.com
- **GCP Project ID:** `ai-lab-16-gcp-510516`
- **Nhánh thực hiện:** Nhánh B — GCP: Terraform + IAP/OS Login
- **Ngày thực hiện:** 04/10/2026

---

## 1. Báo cáo ngắn tóm tắt quá trình thực nghiệm (8 điểm theo yêu cầu CP5)

1. **Hạ tầng triển khai:** Tôi dùng Google Cloud Platform (GCP), region `us-central1` / zone `us-central1-a`, máy ảo CPU `e2-medium` (2 vCPU, 4GB RAM, 30GB SSD Debian 12), source commit `55539f6`.
2. **Dữ liệu:** Dataset Credit Card Fraud có 284,807 dòng (31 cột, 492 gian lận ~0.17%), chia stratified train (72% - 205,060 dòng) / validation (8% - 22,785 dòng) / test (20% - 56,962 dòng), random seed cố định 42.
3. **Hiệu năng huấn luyện:** Load dữ liệu mất 2.953 giây; training LightGBM mất 1.978 giây; best iteration tối ưu là vòng 1 (nhờ early stopping trên tập validation).
4. **Chất lượng mô hình:** AUC-ROC đạt 0.9267, Accuracy đạt 0.9989 (99.89%), F1-Score đạt 0.7382, Precision đạt 0.6370, Recall đạt 0.8776 trên tập test 56,962 mẫu.
5. **Độ trễ và Thông lượng:** Latency 1 dòng trung bình 1.391 ms (đo qua 100 lần lặp sau warm-up); Throughput batch 1,000 dòng đạt 559,602.9 dòng/giây (đo qua 25 batch sau warm-up).
6. **Tài nguyên:** CPU/RAM/Network quan sát ngay sau benchmark: CPU idle 100% (load average 0.04); RAM dùng 507 MiB / 3.8 GiB (còn trống 3.3 GiB khả dụng); Network RX 255.2 MB (tải gói thư viện và dataset qua Cloud NAT) và TX 1.15 MB, 0 dropped packets; ảnh đính kèm trong thư mục `screenshots/`.
7. **Chi phí và Billing:** Billing tại thời điểm 04/10/2026 chưa cập nhật (hiển thị $0.00) do đặc thù độ trễ 12-24 giờ của Google Cloud Billing Reports; ước tính chi phí thực tế cho toàn bộ buổi lab (~40 phút) chỉ khoảng $0.044 (~1.150 VNĐ) và được cấn trừ hoàn toàn vào khoản tín dụng $300 Free Trial.
8. **Dọn dẹp:** Tôi đã tải toàn bộ kết quả (`benchmark.py`, `benchmark_result.json`, log) về laptop và thực hiện lệnh `terraform destroy` xóa sạch 14 tài nguyên GCP; bằng chứng dọn dẹp kiểm tra bằng `terraform state list` trả về trạng thái rỗng.

---

## 2. Bảng Tổng hợp Kết quả Đo lường Chi tiết

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

## 3. Trả lời các câu hỏi kỹ thuật trong Lab

1. **Vì sao AUC-ROC phải dùng xác suất (`predict_proba`) thay vì nhãn cứng (0 hoặc 1)?**
   - *Trả lời:* Đường cong ROC được xây dựng bằng cách tính toán True Positive Rate (TPR) và False Positive Rate (FPR) trên **tất cả các ngưỡng quyết định (threshold) từ 0 đến 1**. Nếu chỉ sử dụng nhãn nhị phân cứng (0 hoặc 1), mô hình sẽ chỉ tạo ra một điểm duy nhất trên không gian ROC, làm mất hoàn toàn khả năng tinh chỉnh ngưỡng cắt (threshold tuning) cho các bài toán dữ liệu mất cân bằng như phát hiện gian lận.

2. **Vì sao Throughput có đơn vị dòng/giây và thời gian inference bao gồm những gì?**
   - *Trả lời:* Throughput đo lường năng lực phục vụ đồng thời của hệ thống trong một đơn vị thời gian (1 giây), tính bằng công thức: `1000 / thời_gian_dự_đoán_1000_dòng (giây)`. Thời gian inference **chỉ tính riêng** thời gian CPU thực hiện việc duyệt cây quyết định để tính toán kết quả đầu ra cho các mẫu dữ liệu, **không bao gồm** thời gian khởi tạo VM, nạp dữ liệu hay huấn luyện mô hình.

---

## 4. Ước tính Chi phí Chi tiết (Region `us-central1`)

| Dịch vụ (Service) | Loại tài nguyên | Chi phí / giờ | Chi phí thực tế buổi lab (~40 phút) |
|---|---|---|---|
| **Compute Engine** | Máy ảo `e2-medium` (2 vCPU, 4GB RAM) | ~$0.033 / giờ | ~$0.022 |
| **Cloud NAT** | Gateway + Egress traffic (~0.26 GB) | ~$0.044 / giờ + data | ~$0.029 |
| **Cloud Load Balancing** | External HTTP LB Forwarding Rule | ~$0.008 / giờ | ~$0.005 |
| **Persistent Disk** | 30 GB Standard SSD Boot Disk | ~$0.002 / giờ | ~$0.001 |
| **TỔNG CỘNG** | | **~$0.087 / giờ** | **~$0.057 (~1.450 VNĐ)** |
