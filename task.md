# Tài Liệu Kỹ Thuật & Theo Dõi Tiến Độ - Project Final DAT301m

Tài liệu này là nguồn sự thật chính cho dự án.
Mục tiêu của nó:

- Giữ lại toàn bộ context kỹ thuật.
- Ghi rõ môi trường, dữ liệu, pipeline, kết quả, lỗi, và quyết định.
- Giúp bất kỳ người nào đọc vào cũng hiểu hiện trạng mà không phải đoán.
- `AGENTS.md` chỉ là file rule vận hành; mọi context chi tiết và trạng thái dự án vẫn phải đọc trong `task.md`.

## 1. Phạm Vi Hiện Tại

- Trọng tâm hiện tại: tinh chỉnh hệ thống nhận diện vật thể bằng YOLOv11.
- GAN chỉ là nhánh lịch sử và sẽ quay lại ở giai đoạn sau.
- Trong nhánh hiện tại, không đụng vào các file GAN `py` nếu chúng không phục vụ YOLO.
- BDD preprocess là dữ liệu tạm, chỉ giữ raw data, config, và log quan trọng.
- Nếu phải chọn giữa giữ dữ liệu và giữ SSD, ưu tiên giữ SSD.

## 2. Nguồn Dữ Liệu Chính Thức

### 2.1 BDD100K

- Kaggle: [solesensei_bdd100k](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
- Bài giới thiệu chính thức: [BDD blog của Berkeley](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
- GitHub toolkit: [bdd-data](https://github.com/ucbdrive/bdd-data)
- BDD100K là bộ dữ liệu lái xe đa miền.
- Trang Kaggle hiện hiển thị:
  - `100k Labeled Road Images | Day, Night`
  - `Version 2 (9.88 GB)`
  - `136k files`

### 2.2 ExDark

- Kaggle: [object-detection-exdark](https://www.kaggle.com/datasets/rociomcomin/object-detection-exdark)
- GitHub chính thức: [ExDark dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- Trang Kaggle hiện hiển thị:
  - `Version 1 (1.51 GB)`
  - `14.7k files`
  - `12 directories`
- ExDark là bộ dữ liệu low-light cho object detection, gồm 12 lớp.

## 3. Môi Trường Và Phần Cứng

### 3.1 Thông số máy thực tế


| Thành phần     | Thông số                         |
| ---------------- | ---------------------------------- |
| CPU              | Intel Core i5-12500H               |
| Core / Thread    | 12 / 16                            |
| RAM              | 24 GB                              |
| GPU              | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM             | 4 GB                               |
| GPU phụ         | Intel Iris Xe Graphics             |
| Hệ điều hành | Windows 11 Home Single Language    |
| Build            | 22631                              |
| Python           | 3.10.11                            |
| PyTorch          | CUDA 12.1                          |
| TensorFlow       | 2.10.0                             |

### 3.2 Hệ quả vận hành

- Máy là Windows native, không phải Linux.
- VRAM 4 GB là nút cổ chai chính.
- Không nên train nhiều job song song trong VSCode.
- Nên giữ model nhỏ, ưu tiên `yolo11n`.
- Nên dùng `batch=8`, `workers=4`, `cache=disk`, `imgsz=640`.
- Với BDD, cache và symlink/copy có thể tạo áp lực lớn lên SSD, nên dữ liệu preprocess chỉ dùng tạm.

## 4. Cấu Trúc Dự Án

```text
D:/DAT301m/proposal/
├── data/
│   ├── raw/
│   │   ├── bdd100k_night/
│   │   └── exdark/
│   ├── exdark.yaml
│   ├── bdd_day.yaml
│   ├── bdd_night.yaml
│   └── bdd100k.yaml        # Legacy, không phải flow chính hiện tại
├── notebooks/
│   ├── 01_data_conversion.ipynb
│   ├── 01b_preprocess_bdd100k.ipynb
│   ├── 02_train_exdark.ipynb
│   ├── 02b_resume_exdark.ipynb
│   ├── 03_train_bdd100k.ipynb
│   └── 03b_resume_bdd100k.ipynb
├── models/
│   └── runs/
├── docs/
├── archive/
├── scripts/                 # Hiện chỉ giữ `setup_exposure_dataset.py`
└── venv/
```

### Ghi chú cấu hình

- File dùng thật cho BDD hiện tại:
  - `data/bdd_day.yaml`
  - `data/bdd_night.yaml`
- `data/bdd100k.yaml` chỉ nên xem là legacy.
- Nếu có notebook cũ còn trỏ `bdd100k.yaml`, coi đó là đường dẫn cũ cần tránh dùng lại.

## 5. Hồ Sơ Dữ Liệu

### 5.1 ExDark

- Nhãn gốc là TXT, theo định dạng bounding box 1-indexed.
- Có 12 lớp:
  - Bicycle
  - Boat
  - Bottle
  - Bus
  - Car
  - Cat
  - Chair
  - Cup
  - Dog
  - Motorbike
  - People
  - Table
- Dữ liệu gốc phân tán theo nhiều thư mục, nên preprocessing phải làm path mapping đệ quy.
- Sau chuẩn hóa, ExDark đã được convert sang YOLO format và dùng để train xong.

### 5.2 BDD100K

- Nhãn gốc là JSON lớn.
- Raw JSON rất nặng nên phải parse kiểu stream, không load nguyên khối vào RAM.
- BDD dùng 10 lớp giao thông:
  - pedestrian
  - rider
  - car
  - truck
  - bus
  - train
  - motorcycle
  - bicycle
  - traffic light
  - traffic sign
- Raw BDD không dùng trực tiếp tên lớp giống YOLO luôn.
- Cần map alias từ JSON gốc:
  - `person` -> `pedestrian`
  - `bike` -> `bicycle`
  - `motor` -> `motorcycle`
- Nếu không map alias, sẽ mất class trong output preprocess.

### 5.3 Chiến Lược Tách Miền

- `daytime` -> `bdd_day`
- `night` -> `bdd_night`
- `dawn/dusk` -> loại bỏ

Lý do:

- Day dùng làm base model.
- Night dùng fine-tune.
- Tách miền giữ được giá trị phân tích của GAN về sau.

### 5.4 Quản Trị Dung Lượng

- ExDark dùng copy vật lý.
- BDD dùng symlink nếu Windows cho phép.
- Nếu symlink lỗi quyền, fallback sang copy.
- BDD preprocess không nên giữ lâu trên SSD.
- Preprocessed BDD nên sinh khi cần train, xong thì xóa.

## 6. Pipeline Tiền Xử Lý

### 6.1 `01_data_conversion.ipynb`

- Chuẩn hóa dữ liệu ExDark sang YOLO format.
- Xử lý tọa độ 1-indexed về 0-indexed.
- Clip tọa độ để tránh out of bounds.
- Tạo dataset cơ sở cho ExDark và baseline CLAHE/Median Filter.

### 6.2 `01b_preprocess_bdd100k.ipynb`

- Parse JSON BDD bằng `ijson`.
- Tách BDD thành `bdd_day` và `bdd_night`.
- Giữ lại occluded boxes để model học sát thực tế.
- Tạo symlink trước, fallback sang physical copy nếu cần.
- BDD preprocess đã được sửa alias mapping và rerun thành công trước khi bị dọn khỏi máy.
- Split cuối đã kiểm tra:
  - `bdd_day`: 36,728 train / 5,258 val
  - `bdd_night`: 27,971 train / 3,929 val

### 6.3 Baseline ảnh tối

- ExDark có baseline `CLAHE + Median Filter`.
- Baseline này phục vụ so sánh với GAN.
- GAN chưa phải đường chạy chính trong nhánh hiện tại.

## 7. Kết Quả Huấn Luyện ExDark

### 7.1 Pha 1: 50 epoch

Kết quả chính:


| Chỉ số | Giá trị |
| -------- | --------- |
| mAP50    | 0.569     |
| mAP50-95 | 0.340     |

### 7.2 Pha 2: resume thêm 80 epoch

Kết quả cuối:


| Chỉ số | Giá trị |
| -------- | --------- |
| mAP50    | 0.636     |
| mAP50-95 | 0.393     |

Nhận định:

- Model đã bão hòa từ khoảng epoch 65.
- Train thêm chỉ tăng rất nhỏ.
- Không đáng để kéo dài thêm vì lợi ích thấp hơn chi phí thời gian.

### 7.3 Hyperparameters chính


| Tham số     | Giá trị |
| ------------ | --------- |
| Model        | YOLO11n   |
| Batch size   | 8         |
| Workers      | 4         |
| Cache        | disk      |
| Image size   | 640       |
| Optimizer    | auto      |
| Patience     | 15        |
| Mosaic       | 0.5       |
| Close mosaic | 10        |
| Seed         | 0         |

### 7.4 Phân tích theo lớp

- Car: rất mạnh.
- Truck: tốt.
- Bus: tốt.
- Traffic light: tốt.
- Traffic sign: tốt.
- Rider: yếu hơn vì dễ dính với pedestrian.
- Train: gần như không học được vì quá ít mẫu.

Kết luận:

- ExDark đã đạt trần hiệu suất hợp lý với `yolo11n`.
- Model này nên được khóa lại làm mốc tham chiếu.

## 8. Kết Quả BDD

### 8.1 Trạng thái trước khi sửa alias

- Có dry run 20 epoch trên `bdd_day`.
- Metrics đã ghi nhận:
  - mAP50: 0.444
  - mAP50-95: 0.261
- Nhưng kết quả này không thể dùng làm kết luận cuối cùng vì lúc đó mapping class BDD còn sai.

### 8.2 Vấn đề đã phát hiện

- Raw BDD dùng `person`, `bike`, `motor`.
- Preprocess cũ chỉ map theo tên lớp YOLO, nên mất ba lớp quan trọng.
- Đây là lỗi logic trong tiền xử lý, không phải lỗi train.

### 8.3 Trạng thái hiện tại

- Preprocess BDD đã được sửa và đã rerun thành công.
- Dữ liệu preprocess BDD đã bị xóa khỏi máy để tiết kiệm SSD.
- Chưa retrain lại BDD trên dataset đã sửa.

### 8.4 Ý nghĩa kỹ thuật

- Khi train lại BDD, phải sinh lại preprocess từ raw.
- Không khôi phục bản preprocess cũ.
- Chỉ dùng `data/bdd_day.yaml` và `data/bdd_night.yaml`.

## 9. GAN

- GAN nằm ngoài nhánh thực thi hiện tại.
- Midterm PDF chỉ là context lịch sử.
- Có một số artifact/evaluation cũ trong repo, nhưng không phải nguồn sự thật cho flow hiện tại.
- Nếu quay lại GAN sau này, cần kiểm tra lại đường dẫn model và dataset vì có dấu hiệu path cũ từ máy khác.

## 10. Lỗi Và Bài Học


| Sự cố                                      | Nguyên nhân                                       | Xử lý                                         |
| -------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| `SyntaxError: accumulate=8`                  | Ultralytics đã bỏ tham số`accumulate` khỏi API | Xóa tham số, để auto-accumulate mặc định |
| `yolo26n.pt` xuất hiện trong notebooks     | File nháp từ AMP check                            | Đã xóa                                       |
| RAM tăng cao khi dùng`cache=disk`          | Windows standby cache + workers                     | Chấp nhận như đặc tính vận hành         |
| Resume chạy dài hơn dự kiến             | Dùng`resume=True` sai ngữ cảnh                   | Dùng warm start thay vì cố resume cứng      |
| Mất class BDD                               | Sai mapping alias raw labels                        | Sửa preprocess và rerun                       |
| `bdd100k.yaml` bị lẫn vào flow hiện tại | Legacy config                                       | Chỉ giữ làm tài liệu cũ                   |

## 11. Checklist Tiến Độ

### 11.1 Hoàn thành

- [X]  Khởi tạo cấu trúc repo.
- [X]  Thiết lập môi trường `venv`.
- [X]  Cài PyTorch CUDA 12.1, TensorFlow 2.10, Ultralytics.
- [X]  Chuẩn hóa ExDark sang YOLO format.
- [X]  Tạo baseline CLAHE/Median Filter.
- [X]  Thiết kế và chạy preprocess BDD theo tách miền day/night.
- [X]  Sửa alias mapping BDD raw labels.
- [X]  Train ExDark 50 epoch.
- [X]  Resume ExDark thêm 80 epoch.
- [X]  Chốt ExDark là mốc baseline chính thức.
- [X]  Dọn BDD preprocess khỏi SSD sau khi hoàn tất kiểm tra.

### 11.2 Đang mở

- [ ]  Base train BDD day trên dataset đã sửa.
- [ ]  Fine-tune BDD night từ base.
- [ ]  Đánh giá COCO mAP50-95 cho BDD sau retrain.
- [ ]  So sánh Raw vs CLAHE vs GAN nếu quay lại nhánh tăng cường ảnh.

### 11.3 Pending

- [ ]  GAN inference pipeline.
- [ ]  Đánh giá ảnh tăng cường.
- [ ]  Tổng hợp biểu đồ so sánh cuối cùng cho báo cáo.

## 12. Dung Lượng Và Lưu Trữ

### Giữ lại

- `data/raw/`
- `data/bdd_day.yaml`
- `data/bdd_night.yaml`
- `data/exdark.yaml`
- `models/runs/` quan trọng
- `archive/models/runs/bdd_day_*`

### Không giữ lâu

- `data/processed/bdd_day`
- `data/processed/bdd_night`
- `labels.cache`
- `*.npy`
- các dataset tạm sinh ra chỉ để validate

### Quy tắc

- Preprocess chỉ sinh khi cần train.
- Train xong thì dọn cache nếu cần giải phóng SSD.
- Không giữ bản preprocess BDD cũ trên máy.

## 13. Trạng Thái Hiện Tại

- Máy đang chạy Windows, VSCode, GPU 4 GB VRAM.
- ExDark đã xong và có thể dùng làm mốc.
- BDD preprocess đã sửa nhưng đã xóa khỏi máy để tiết kiệm dung lượng.
- BDD train chưa làm lại trên dataset đã sửa.
- GAN vẫn là nhánh sau.
- Tài liệu này là nguồn đọc nhanh cho bất kỳ AI nào vào dự án.
