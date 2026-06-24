# Tài Liệu Kỹ Thuật & Theo Dõi Tiến Độ (Project Final DAT301m)

Tài liệu này đóng vai trò cơ sở định hướng cho toàn bộ dự án, bao gồm: Lộ trình triển khai, hồ sơ cấu trúc phần cứng/dữ liệu, đánh giá rủi ro kỹ thuật chuyên sâu và bảng theo dõi tiến độ (Checklist).

_Lưu ý: Hiện tại dự án tập trung vào việc tinh chỉnh hệ thống Nhận diện vật thể (Object Detection - YOLOv11). Phân hệ Tăng cường ảnh bằng Mạng đối nghịch sinh (GAN) được xếp lịch vào Giai đoạn 4._

---

## 1. Nguồn Dữ Liệu (Official Dataset Links)

Do dung lượng các tập dữ liệu lớn và yêu cầu xác thực, cần tiến hành tải thủ công từ các nguồn chính thức sau:

1. **BDD100k Dataset**:
   - Link Kaggle: [solesensei_bdd100k](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
   - Lệnh Kaggle API: `kaggle datasets download -d solesensei/solesensei_bdd100k`
   - _Ghi chú_: Repository này chứa toàn bộ 100K ảnh gốc và tệp nhãn `.json` đúng chuẩn của Berkeley.
2. **ExDark (Exclusively Dark) Dataset**:
   - Link Kaggle: [object-detection-exdark](https://www.kaggle.com/datasets/rociomcomin/object-detection-exdark)
   - Lệnh Kaggle API: `kaggle datasets download -d rociomcomin/object-detection-exdark`
   - _Ghi chú_: Repository chứa 7,363 ảnh và 7,363 tệp nhãn gốc. Có sự sai lệch trong cấu trúc dữ liệu của tác giả:
     - Thư mục nhãn bị gõ sai chính tả thành `ExDark_Annno`.
     - Tệp nhãn giữ nguyên phần mở rộng của ảnh (ví dụ: `2015_00001.png.txt`) và phân tán trong 12 thư mục con.
     - **Giải pháp**: Tích hợp thuật toán đệ quy trong `01_data_conversion.ipynb` nhằm tự động chuẩn hóa đường dẫn và tên tệp.

---

## 2. Hồ Sơ Phần Cứng (Hardware Specifications)

Kiến trúc mạng Neural và các giải thuật tối ưu trong dự án này được thiết kế và khóa cứng dựa trên cấu hình phần cứng vật lý khả dụng của trạm xử lý:

| Thành phần       | Thông số                                                |
| ---------------- | ------------------------------------------------------- |
| **GPU**          | NVIDIA GeForce RTX 3050 Laptop GPU                      |
| **VRAM**         | 4.0 GB*(Ràng buộc nghiêm ngặt nhất)*                    |
| **System RAM**   | ~24 GB (23.71 GB khả dụng)                              |
| **Ổ cứng**       | SSD NVMe                                                |
| **Hệ Điều Hành** | Windows (Native)                                        |
| **Python**       | 3.10.11                                                 |
| **PyTorch**      | CUDA 12.1 — Xử lý mạng YOLOv11                          |
| **TensorFlow**   | 2.10.0 — Phiên bản cuối hỗ trợ GPU Windows, phục vụ GAN |

_Khuyến nghị Học thuật: Mọi cấu hình siêu tham số (Batch Size, Accumulate, Model Size) trong các giai đoạn sau đều được nội suy và thiết lập theo tỷ lệ nghịch với dung lượng 4GB VRAM của card RTX 3050 nhằm triệt tiêu lỗi Out of Memory (OOM)._

---

## 3. Cấu Trúc Thư Mục Dự Án (Project Directory Structure)

```text
D:/DAT301m/proposal/
├── data/
│   ├── raw/                     # Dữ liệu gốc, không chỉnh sửa.
│   │   ├── bdd100k_night/       # Ảnh và JSON gốc BDD100k.
│   │   └── exdark/              # Ảnh và TXT gốc ExDark.
│   ├── processed/               # Dữ liệu đã chuẩn hóa sang YOLO format.
│   │   ├── train/images/        # 5,890 ảnh ExDark (train)
│   │   ├── train/labels/        # 5,890 nhãn ExDark (train)
│   │   ├── val/images/          # 1,472 ảnh ExDark (val)
│   │   └── val/labels/          # 1,472 nhãn ExDark (val)
│   ├── clahe_baseline/          # Ảnh tối qua CLAHE/Median Filter (11,780 train + 2,944 val).
│   ├── enhanced/                # [Pending] Ảnh sinh từ GAN.
│   ├── exdark.yaml              # Cấu hình YOLO cho ExDark (12 classes).
│   └── bdd100k.yaml             # Cấu hình YOLO cho BDD100k (10 classes).
├── notebooks/
│   ├── 01_data_conversion.ipynb # Tiền xử lý, chuẩn hóa tọa độ, phân tách.
│   ├── 02_train_exdark.ipynb    # Huấn luyện ExDark (Pha 1: 50 Epochs).
│   ├── 02b_resume_exdark.ipynb  # Resume ExDark (+80 Epochs từ trọng số Epoch 50).
│   └── 03_train_bdd100k.ipynb   # Huấn luyện BDD100k (Chờ xử lý).
├── models/
│   ├── pretrained/              # Trọng số khởi tạo (yolo11n.pt).
│   └── runs/exdark_train-2/     # Kết quả huấn luyện ExDark (đang chạy).
│       ├── weights/             # best.pt, last.pt, best_epoch50.pt, last_epoch50.pt
│       ├── milestone_epoch50/   # Sao lưu mốc lịch sử Pha 1.
│       └── results.csv          # Log huấn luyện theo epoch.
├── docs/                        # Tài liệu tham khảo, Proposal, PPTX.
├── scripts/                     # Mã nguồn tự động hóa sinh Notebook.
└── venv/                        # Môi trường ảo Python (PyTorch CUDA 12.1).
```

---

## 4. Hồ Sơ Cấu Trúc Dữ Liệu (Dataset Profiling)

Để đảm bảo tính toàn vẹn trong quá trình huấn luyện, hai tập dữ liệu đã được tiến hành phân tích và lập hồ sơ cấu trúc chi tiết:

### 4.1. Tập Dữ Liệu ExDark (Exclusively Dark)

- **Định dạng nhãn gốc**: Tệp văn bản (TXT) theo định dạng danh sách tọa độ và lớp.
- **Hệ tọa độ**: 1-indexed (lệch 1 pixel so với không gian ma trận 0-indexed của Python/PyTorch), yêu cầu dịch chỉnh tọa độ trong quá trình chuẩn hóa YOLO.
- **Cơ cấu phân lớp (Classes)**: Phân bổ thành 12 lớp (Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table). Có sự xen lẫn các vật thể sinh hoạt phi giao thông.
- **Khối lượng dữ liệu**: 7,363 ảnh đặc tả môi trường thiếu sáng. Phân bổ Validation theo tỷ lệ động 80:20.
- **Dị biệt cấu trúc**: Tên tệp nhãn bảo lưu phần mở rộng của ảnh (`.png.txt`, `.jpg.txt`), đòi hỏi thuật toán ghép chuỗi phức hợp để truy xuất.

### 4.2. Tập Dữ Liệu BDD100k (Berkeley DeepDrive)

- **Định dạng nhãn gốc**: Tệp JSON nguyên khối (Monolithic JSON) dung lượng lớn (1.4GB+), cấu trúc Nested Dictionary.
- **Chiến lược Phân tích (Parsing)**: Bắt buộc sử dụng kỹ thuật luồng (stream parsing) qua thư viện `ijson` để tránh tràn RAM (OOM) khi nạp file 1.4GB.
- **Hệ tọa độ**: 0-indexed, cấu trúc Box2D (`x1, y1, x2, y2`) chuẩn xác trên độ phân giải 1280x720. Cần chuẩn hóa về `[x_center, y_center, w, h]` trong khoảng `[0, 1]` cho YOLO.
- **Cơ cấu phân lớp (Classes)**: Giới hạn chuyên sâu trong 10 lớp giao thông tự hành (Pedestrian, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Traffic light, Traffic sign).
- **Phân loại môi trường & Tách Miền (Domain Split)**:
  - Ảnh `daytime` (~52k): Được trích xuất thành miền `bdd_day`. Sử dụng làm nền tảng huấn luyện cơ sở (Base Training) cho YOLO và làm Ground Truth cho GAN.
  - Ảnh `night` (~39k): Được trích xuất thành miền `bdd_night`. Nơi tiến hành Fine-tune chuyên sâu.
  - Ảnh `dawn/dusk`: Bị loại bỏ hoàn toàn để tránh nhiễu tranh tối tranh sáng.
- **Kiểm soát nhiễu**: Loại bỏ các Box có cờ `occluded=True` nhằm giảm thiểu dao động trong hàm Loss.

### 4.4. Chiến Lược Quản Trị Dung Lượng (Symlink vs Physical Copy)

Do sự chênh lệch khổng lồ về quy mô (ExDark ~7k ảnh vs BDD100k ~100k ảnh), dự án áp dụng 2 chiến lược quản lý file khác nhau:

1. **ExDark**: Sử dụng `shutil.copy()` (Physical Copy). Tốn thêm vài trăm MB, tốc độ cực nhanh, an toàn tuyệt đối khỏi các lỗi phân quyền HĐH. Phù hợp vì sau đó cần áp dụng bộ lọc CLAHE lên file ảnh copy.
2. **BDD100k**: Sử dụng `os.symlink()` (Symbolic Link). YOLO sẽ đọc "lối tắt" trỏ về file Raw thay vì copy vật lý, giúp tiết kiệm triệt để ~15GB SSD. Không cần resize ảnh vì YOLO hỗ trợ _Dynamic Letterboxing_.
   - _Cơ chế Fallback (Tự chữa cháy)_: Trên Windows, tạo Symlink đòi hỏi quyền Administrator/Developer Mode. Script `preprocess_bdd100k.py` được thiết kế thông minh: cố gắng tạo Symlink, nếu HĐH báo lỗi PermissionError thì tự động chuyển sang Physical Copy để đảm bảo không đứt gãy luồng thực thi.

### 4.3. Kiểm kê Dữ liệu Thực tế (Data Inventory)

| Thư mục                | Train                  | Val                    | Ghi chú                              |
| ---------------------- | ---------------------- | ---------------------- | ------------------------------------ |
| `processed/` (ExDark)  | 5,890 ảnh + 5,890 nhãn | 1,472 ảnh + 1,472 nhãn | Đang sử dụng                         |
| `processed/` (BDD100k) | ** CHƯA CÓ**           | ** CHƯA CÓ**           | Cần chạy lại Data Conversion cho BDD |
| `clahe_baseline/`      | 11,780 files           | 2,944 files            | Sẵn sàng (ExDark + nhãn)             |
| `enhanced/` (GAN)      | —                      | —                      | Pending Giai đoạn 4                  |

---

## 5. Lộ Trình Triển Khai & Phân Tích Kỹ Thuật (Roadmap)

### `01_data_conversion.ipynb` (Chuẩn hóa & Tiền xử lý Data)

- **Mục tiêu**: Chuyển đổi dữ liệu thô sang định dạng nhãn YOLO (0-indexed).
- **Xử lý tập ExDark**: Tọa độ nguyên bản là 1-indexed, cần được dịch chỉnh về 0-indexed. Loại bỏ các header không hợp lệ (`% bbGt`). Áp dụng hàm `np.clip(tọa_độ, 0, 1)` để ngăn ngừa lỗi tọa độ vượt biên (Out of bounds) gây ra hiện tượng Loss = NaN trong quá trình huấn luyện.

### `scripts/preprocess_bdd100k.py` (Tiền xử lý BDD100k - Tách Miền/Domain Split)

- **Vấn đề**: BDD100k có 1.4GB JSON và 100k ảnh. Nếu gộp chung Day/Night, YOLO sẽ học rất tốt nhưng bị "bias" (thiên lệch) về ảnh ban ngày, làm mất đi giá trị của mô-đun GAN ở Giai đoạn 4.
- **Giải pháp**: Phân tách BDD100k thành 2 tập độc lập `bdd_day` và `bdd_night` (loại bỏ hoàn toàn thẻ thời tiết lấp lửng `dawn/dusk`).
- **Xử lý luồng JSON**: Sử dụng thư viện `ijson` để parse từng node một thay vì load toàn bộ vào RAM, giúp mức sử dụng RAM duy trì ở mức phẳng (<1GB). Lọc bỏ các Box `occluded=True` và chuẩn hóa tọa độ về YOLO format `[0, 1]`.
- **Chiến lược Domain Adaptation (Phương án 3)**:
  1. **Base Training**: Huấn luyện YOLO trên tập `bdd_day` trước. Mô hình sẽ học được cấu trúc hình học sắc nét của xe cộ, biển báo dưới ánh sáng lý tưởng.
  2. **Nighttime Fine-tuning**: Khóa các lớp đặc trưng (hoặc dùng Learning Rate thấp), đem trọng số đã học ở trên fine-tune độc quyền trên tập `bdd_night`. Mô hình trở thành "Chuyên gia bóng đêm có nền tảng học vấn từ ban ngày". Chiến lược này cho mAP cao hơn hẳn việc chỉ train ảnh đêm từ số 0, và đồng thời giữ nguyên đất diễn cho GAN tỏa sáng ở Giai đoạn 4.
- **Chiến lược Tiền xử lý Không gian (Spatial Preprocessing)**: Quyết định bảo lưu độ phân giải nguyên bản của ảnh trên ổ cứng thay vì thu nhỏ (Hard Resize). Hệ thống phó thác hoàn toàn cơ chế thay đổi kích thước cho hàm DataLoader của YOLO thông qua kỹ thuật **Dynamic Letterboxing** (`imgsz=640`). Điều này nhằm bảo toàn trọn vẹn chất lượng quang học của ảnh gốc, phục vụ các bài kiểm tra chéo (cross-validation) ở đa dạng phân giải trong tương lai.
- **Khởi tạo Baseline Truyền thống**: Tích hợp Median Filter (giảm nhiễu ISO) và CLAHE (cân bằng lược đồ xám) để tạo tập `clahe_baseline`. Sử dụng Letterbox Padding của YOLO nhằm duy trì tỷ lệ khung hình (Aspect Ratio).

### `02_train_exdark.ipynb` & `03_train_bdd100k.ipynb` (Huấn luyện Mô hình YOLOv11)

- **Mục tiêu**: Huấn luyện và so sánh mAP của kiến trúc YOLOv11 trên các miền dữ liệu (Tối nguyên bản và Tối qua xử lý CLAHE).
- **Thiết kế Độc Lập**: Triển khai huấn luyện 2 mô hình YOLO biệt lập cho ExDark (12 lớp) và BDD100k (10 lớp) để loại trừ xung đột không gian nhãn.
- **Tối ưu hóa Tài nguyên (4GB VRAM Constraint)**: Bắt buộc triển khai kiến trúc **YOLO11n (Nano)** với ~2.6 triệu tham số. Khóa cứng kích thước vật lý `batch_size=8` để tránh tràn bộ nhớ (Out of Memory). Hệ thống tự động nội suy kỹ thuật **Auto-Accumulate** (64 / Batch = 8) nhằm mô phỏng kích thước lô ảo lên 64, giúp ổn định đạo hàm mà không vi phạm cú pháp API mới của Ultralytics.
- **Quản trị Khủng hoảng System RAM (Windows Standby Cache & Multi-processing)**:
  - _Nút thắt cổ chai (Bottleneck)_: Trong môi trường Windows, thư viện PyTorch kích hoạt đa luồng (multi-processing) thông qua cơ chế `spawn` (tạo mới quy trình) thay vì `fork` (sao chép quy trình). Tham số mặc định `workers=8` yêu cầu cấp phát 8 phiên bản bộ nhớ độc lập cho DataLoader, làm bùng nổ xung đột không gian nhớ.
  - _Cơ chế Caching Ổ cứng (`cache='disk'`)_: Thay vì để CPU liên tục giải nén ảnh JPEG và nội suy kích thước (Dynamic Letterboxing) theo thời gian thực (gây quá tải CPU), hệ thống được lệnh tiền xử lý toàn bộ tập dữ liệu thành các ma trận NumPy thuần túy (uncompressed tensors) và ghi thẳng xuống SSD. Quá trình này tiêu tốn 12.5 GB dung lượng vật lý, nhưng triệt tiêu hoàn toàn độ trễ giải nén.
  - _Hiện tượng Ảo ảnh Bộ nhớ (Memory Mirage)_: Khi đọc liên tục file `.cache` 12.5 GB từ SSD, Windows OS Memory Manager tự động thu hồi toàn bộ RAM nhàn rỗi để đưa vào danh sách chờ (`Standby List`), biến RAM thành một bộ đệm siêu tốc (Disk Cache). Dù Task Manager báo động đỏ System RAM chạm mức 95%+, thực tế đây chỉ là "RAM Ảo". Lượng RAM này sẽ lập tức được hệ điều hành giải phóng (evicted) nếu có tiến trình ưu tiên cao hơn yêu cầu cấp phát. Đây là hành vi thiết kế có chủ đích (By Design) và an toàn tuyệt đối.
- **Cơ chế Chống Quá Khớp (Anti-Overfitting & Hallucination)**:
  - Áp dụng **Early Stopping** (`patience=15`), tự động ngắt chu kỳ huấn luyện nếu mAP không cải thiện.
  - Sử dụng **Mosaic Augmentation** (`mosaic=0.5`) để tăng cường bối cảnh không gian, đi kèm `close_mosaic=10` ở các epoch cuối cùng nhằm tinh chỉnh viền đối tượng nguyên bản.
  - Kích hoạt trình tối ưu **AdamW** với cơ chế Weight Decay để kiểm soát ngưỡng bùng nổ tham số.

### `02_gan_inference.ipynb` (Tạm Gác / Pending)

- **Mục tiêu**: Áp dụng mạng U-Net GAN để tăng cường độ sáng ảnh đầu vào.
- **Quản trị Tài nguyên VRAM**: Thiết lập `batch_size=1`, kích hoạt `tf.config.experimental.set_memory_growth`. Áp dụng cơ chế Checkpoint Resume định kỳ nhằm đảm bảo an toàn dữ liệu khi xảy ra hiện tượng Memory Leak.
- **Tính tương thích**: Yêu cầu môi trường TensorFlow 2.10 trên nền Python 3.10 và `numpy<2.0.0` để duy trì hỗ trợ GPU nguyên bản trên Windows.

---

## 6. Kết Quả Huấn Luyện ExDark — Pha 1 (Epoch 1–50)

### 6.1. Bảng Siêu Tham Số Huấn Luyện

| Tham số         | Giá trị        | Lý do                                           |
| --------------- | -------------- | ----------------------------------------------- |
| Model           | YOLO11n (Nano) | Kiến trúc nhẹ nhất (~2.6M params), vừa VRAM 4GB |
| Batch Size      | 8              | Mức tối đa VRAM 4GB chịu được                   |
| Auto-Accumulate | 64/8 = 8       | Mô phỏng batch ảo = 64 để ổn định đạo hàm       |
| Image Size      | 640×640        | Dynamic Letterboxing giữ tỷ lệ khung hình       |
| Optimizer       | AdamW          | Chuyên biệt cho tập dữ liệu nhiễu hạt           |
| Epochs          | 50 (Pha 1)     | Early Stopping không kích hoạt → chưa bão hòa   |
| Patience        | 15             | Dừng sớm nếu mAP không cải thiện                |
| Mosaic          | 0.5            | Augmentation ghép 4 ảnh ngẫu nhiên              |
| Close Mosaic    | 10             | Tắt Mosaic ở 10 epoch cuối để học viền thật     |
| Cache           | disk (12.5 GB) | Giảm tải CPU giải nén, đánh đổi dung lượng SSD  |
| AMP             | True           | Mixed Precision FP16 tăng tốc                   |

### 6.2. Bảng Kết Quả Chính (Epoch 50)

| Chỉ số       | Giá trị                 | Đánh giá                         |
| ------------ | ----------------------- | -------------------------------- |
| **mAP50**    | **0.569** (56.9%)       | Trung bình khá cho điều kiện tối |
| **mAP50-95** | **0.340** (34.0%)       | Tiêu chuẩn COCO, chấp nhận được  |
| Precision    | 0.613                   | 61.3% dự đoán đúng               |
| Recall       | 0.528                   | 52.8% vật thể được phát hiện     |
| Train Time   | ~7,230 giây (~120 phút) | 2 giờ trên RTX 3050              |
| VRAM sử dụng | ~1.35G / 4.0G           | Còn headroom an toàn             |

### 6.3. Bảng mAP50 Theo Từng Lớp (Per-Class Performance)

| Lớp       | mAP50     | Nhận xét                            |
| --------- | --------- | ----------------------------------- |
| Bus       | **0.817** | Tốt nhất — vật thể to, hình dạng rõ |
| Bicycle   | 0.680     | Tốt                                 |
| Car       | 0.679     | Tốt                                 |
| People    | 0.671     | Tốt — xuất hiện nhiều nhất          |
| Motorbike | 0.597     | Khá                                 |
| Bottle    | 0.581     | Khá                                 |
| Boat      | 0.559     | Trung bình                          |
| Cup       | 0.512     | Trung bình                          |
| Dog       | 0.508     | Trung bình — dễ lẫn với Cat         |
| Cat       | 0.438     | Yếu — hình dạng tương tự Dog        |
| Chair     | 0.431     | Yếu — viền mờ trong bóng tối        |
| Table     | **0.357** | Yếu nhất — phẳng, ít đặc trưng      |

### 6.4. Xu hướng Huấn luyện (Learning Curve Snapshot)

| Epoch | box_loss | cls_loss | mAP50 | mAP50-95 |
| ----- | -------- | -------- | ----- | -------- |
| 1     | 2.198    | 3.589    | 0.009 | 0.003    |
| 10    | 1.805    | 2.577    | 0.252 | 0.128    |
| 20    | 1.676    | 2.205    | 0.402 | 0.221    |
| 30    | 1.574    | 1.930    | 0.469 | 0.267    |
| 40    | 1.489    | 1.724    | 0.526 | 0.312    |
| 50    | 1.385    | 1.437    | 0.569 | 0.340    |

_Nhận xét_: Cả 3 hàm Loss đều giảm đều, mAP tăng liên tục không có dấu hiệu bão hòa (plateau) → Quyết định Resume thêm 80 Epochs.

---

## 7. Nhật Ký Sự Cố & Bài Học Kinh Nghiệm (Incident Log)

| #   | Sự cố                                     | Nguyên nhân                                     | Giải pháp                                                                                                                          |
| --- | ----------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `SyntaxError: accumulate=8`               | Ultralytics đã xóa tham số`accumulate` khỏi API | Xóa tham số, để Auto-Accumulate ngầm định                                                                                          |
| 2   | `yolo26n.pt` xuất hiện trong notebooks/   | AMP Check tự tải model nháp từ GitHub           | Xóa thủ công (file rác, 5.5MB)                                                                                                     |
| 3   | RAM 95%+ dù dùng`cache='disk'`            | Windows Standby Cache + 8 workers spawn         | Hiện tượng By Design, an toàn, kệ                                                                                                  |
| 4   | Resume Training chạy 80 epoch thay vì +30 | Truyền params thừa cùng`resume=True`            | Cú pháp đúng: chỉ cần`model.train(resume=True)` không truyền gì thêm. Chấp nhận để chạy tiếp 80 epoch (tổng ~130 epoch kiến thức). |
| 5   | `results.csv` bị xóa khi Resume           | YOLO reset file khi nhận`resume` sai cách       | Khôi phục từ bộ nhớ agent →`milestone_epoch50/results_epoch50.csv`                                                                 |

---

## 8. Bảng Theo Dõi Tiến Độ (Task Tracker Checklist)

- `[x]` **Giai đoạn 1: Khởi tạo Project & Môi trường (Environment Setup)**
  - `[x]` Thiết lập cấu trúc thư mục quy chuẩn (`data`, `notebooks`, `models`, `docs`, `scripts`).
  - `[x]` Khởi tạo môi trường ảo `venv` và cài đặt Jupyter Kernel (`project final DAT301m`).
  - `[x]` Cài đặt thư viện cốt lõi: PyTorch CUDA 12.1, TensorFlow 2.10 (Numpy < 2), Ultralytics YOLO.
  - `[x]` Tổ chức tài liệu tham khảo vào thư mục `docs/`.

- `[x]` **Giai đoạn 2: Convert Data & Tiền xử lý**
  - `[x]` Khởi tạo Notebook `01_data_conversion.ipynb`.
  - `[x]` Xây dựng thuật toán phân tích XML/txt cho ExDark, cấu hình hàm cắt viền (clip) giới hạn tọa độ.
  - `[x]` Xây dựng quy trình tạo tập dữ liệu cơ sở `CLAHE + Median Filter`.
  - `[x]` Di dời dữ liệu BDD100k vào cấu trúc thư mục tiêu chuẩn.
  - `[x]` Xây dựng kịch bản xử lý luồng JSON cho BDD100k (ijson stream), phân tách ảnh `night` và lọc nhiễu `occluded`.
  - `[x]` Lưu trữ bảo lưu dữ liệu ban ngày của BDD100k làm Domain Mẫu (Ground Truth) cho pha huấn luyện GAN.
  - `[x]` **Hoàn tất**: Đã thực thi thành công toàn bộ Notebook `01_data_conversion.ipynb` và nghiệm thu dữ liệu đầu ra.

- `[/]` **Giai đoạn 3: Huấn luyện & Đánh giá YOLOv11 (2 Model Độc Lập)**
  - `[x]` Quyết định thiết kế: Huấn luyện 2 mô hình YOLO độc lập (ExDark 12 classes, BDD100k 10 classes). Hủy bỏ Auto-Evolve để giảm tải tính toán.
  - `[x]` Khởi tạo thành công Notebook `02_train_exdark.ipynb` và `03_train_bdd100k.ipynb`.
  - `[x]` Cấu hình siêu tham số chống tràn RAM: YOLO11n, Batch=8, Auto-Accumulate, Cache=Disk.
  - `[x]` Cấu hình cơ chế chống Overfit: Patience=15, Mosaic=0.5, Close_Mosaic=10, AdamW.
  - `[x]` **Hoàn tất Pha 1**: ExDark 50 Epochs → mAP50=0.569, mAP50-95=0.340. Sao lưu mốc `milestone_epoch50/`.
  - `[/]` **Đang chạy Pha 2**: ExDark +80 Epochs từ trọng số Epoch 50 (tổng ~130 epoch kiến thức). Notebook: `02b_resume_exdark.ipynb`.
  - `[x]` **Chuẩn bị BDD100k**: Đã viết thành công `scripts/preprocess_bdd100k.py` với cơ chế ijson stream và Auto-Fallback Symlink/Copy.
  - `[ ]` **⚠ ĐANG CHỜ**: Chờ ExDark train xong 80 Epoch mới được phép chạy script tiền xử lý BDD100k để tránh nghẽn I/O ổ cứng.
  - `[ ]` Huấn luyện mô hình YOLO11n trên tập Ban ngày (`bdd_day`) để tạo Base Model.
  - `[ ]` Fine-tune mô hình YOLO11n trên tập Ban đêm (`bdd_night`) từ trọng số Base.
  - `[ ]` Đánh giá mAP theo tiêu chuẩn COCO (mAP50-95).
  - `[ ]` Đánh giá mAP trên tập Tiền xử lý (`CLAHE`).
  - `[ ]` Trực quan hóa dữ liệu mAP và xuất báo cáo.

- `[ ]` **Giai đoạn 4: Inference GAN (Đang Pending)**
  - `[x]` Thiết lập cấu hình môi trường TF 2.10 + Numpy < 2.0 chuyên biệt (Đã thực hiện tại GĐ1).
  - `[ ]` Triển khai quy trình Inference xử lý ảnh chống tràn VRAM.
  - `[ ]` Đánh giá mAP của tập ảnh Tăng cường (`Enhanced`).
  - `[ ]` Xây dựng biểu đồ so sánh tổng hợp: Raw vs CLAHE vs GAN.

---

## 9. Tệp Rác Cần Dọn Dẹp (Cleanup)

| File                | Vị trí                           | Lý do                                                |
| ------------------- | -------------------------------- | ---------------------------------------------------- |
| `yolo26n.pt`        | `notebooks/`                     | File nháp AMP Check, 5.5MB rác                       |
| `yolo11n.pt`        | `notebooks/`                     | Bản sao trọng số COCO, nên nằm ở`models/pretrained/` |
| `exdark_train` (cũ) | `models/runs/`                   | Kết quả lần train lỗi đầu tiên (nếu còn)             |
| `labels.cache`      | `data/processed/train/` & `val/` | File cache YOLO, xóa được sau khi train xong         |
