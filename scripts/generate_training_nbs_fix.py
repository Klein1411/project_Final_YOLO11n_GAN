import json
import os

exdark_yaml = """path: d:/DAT301m/proposal/data/processed
train: train/images
val: val/images

names:
  0: bicycle
  1: boat
  2: bottle
  3: bus
  4: car
  5: cat
  6: chair
  7: cup
  8: dog
  9: motorbike
  10: people
  11: table
"""
with open(r'd:\DAT301m\proposal\data\exdark.yaml', 'w') as f:
    f.write(exdark_yaml)

bdd_yaml = """path: d:/DAT301m/proposal/data/processed
train: train/bdd_images
val: val/bdd_images

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
"""
with open(r'd:\DAT301m\proposal\data\bdd100k.yaml', 'w') as f:
    f.write(bdd_yaml)

def create_notebook(filename, title, yaml_name):
    name_str = title.lower().replace(" ", "_")
    nb = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        f"# Giai đoạn 3: Huấn luyện Kiến trúc YOLOv11 ({title})\n",
        "Tài liệu này thực thi quy trình định chuẩn:\n",
        "1. Khởi tạo và thiết lập môi trường thư viện YOLOv11 tiên tiến nhất.\n",
        "2. Kiểm định tài nguyên phần cứng (Khả năng đáp ứng của GPU Cuda).\n",
        "3. Kích hoạt kỹ thuật Tích lũy Gradient (Gradient Accumulation) nhằm vượt qua rào cản VRAM vật lý, kết hợp cơ chế kiểm soát Overfitting."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Khởi tạo thư viện cốt lõi\n",
        "import sys\n",
        "!{sys.executable} -m pip install -U ultralytics\n",
        "!{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Kiểm định tính khả dụng của phần cứng GPU\n",
        "import torch\n",
        "if torch.cuda.is_available():\n",
        "    print(f\"Hệ thống nhận diện phần cứng xử lý: {torch.cuda.get_device_name(0)}\")\n",
        "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
        "    print(f\"Dung lượng VRAM khả dụng thực tế: {vram:.2f} GB\")\n",
        "else:\n",
        "    print(\"Lỗi nghiêm trọng: Không phát hiện nhân tính toán CUDA. Quá trình phân tích sẽ bị gián đoạn.\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Kích hoạt Mô hình YOLOv11 Nano (YOLO11n)\n",
        "Áp dụng cấu hình phân rã khối lượng công việc nhằm bảo đảm an toàn VRAM (4GB):\n",
        "- **Kích thước vật lý (Batch Size)** = 8 (Mức độ dung nạp tối đa của VRAM 4GB).\n",
        "- **Hệ số giả lập Gradient tự động (Auto-Accumulate)**: Thư viện Ultralytics tự động thiết lập Accumulate = 64 / Batch. Với Batch=8, hệ thống sẽ ngầm định tích lũy ma trận sai số qua 8 chu kỳ trước khi tối ưu (8 x 8 = 64), giúp ổn định đạo hàm mà không cần code tay dài dòng.\n",
        "- **Cơ chế Chống Overfitting**: Áp dụng Early Stopping (`patience=15`), kết hợp ngắt Augmentation (`close_mosaic=10`) ở các chu kỳ cuối nhằm tối ưu hóa độ sắc nét của viền đối tượng."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "from ultralytics import YOLO\n",
        "\n",
        "# Tải kiến trúc mạng YOLO11n trọng số khởi tạo chuẩn\n",
        "model = YOLO('yolo11n.pt')\n",
        "\n",
        "# Khởi động quá trình Huấn luyện\n",
        "results = model.train(\n",
        f"    data='d:/DAT301m/proposal/data/{yaml_name}',\n",
        "    epochs=50,\n",
        "    patience=15,           # Dừng sớm nếu sau 15 epoch mAP không tăng (Chống Overfit)\n",
        "    batch=8,               # Kích thước khung nạp vật lý\n",
        "    cache='disk',          # Bộ đệm ổ cứng giảm tải System RAM\n",
        "    imgsz=640,\n",
        "    device=0,\n",
        "    optimizer='AdamW',     # Giải thuật tối ưu chuyên biệt cho tập nhiễu hạt\n",
        "    mosaic=0.5,            # Bật Data Augmentation ghép 4 ảnh\n",
        "    close_mosaic=10,       # Tắt ghép ảnh ở 10 epoch cuối để học viền thực tế\n",
        "    project='d:/DAT301m/proposal/models/runs',\n",
        f"    name='{name_str}_train'\n",
        ")"
       ]
      }
     ],
     "metadata": {
      "kernelspec": {
       "display_name": "project final DAT301m",
       "language": "python",
       "name": "project_final_dat301m"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 5
    }
    nb_path = f'd:/DAT301m/proposal/notebooks/{filename}'
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

create_notebook('02_train_exdark.ipynb', 'ExDark', 'exdark.yaml')
create_notebook('03_train_bdd100k.ipynb', 'BDD100k', 'bdd100k.yaml')
print("Hoan thanh.")
