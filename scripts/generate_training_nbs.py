import json
import os
from pathlib import Path

# Cấu trúc YAML cho ExDark
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

# Cấu trúc YAML cho BDD100k
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
        "3. Kích hoạt giải thuật Tiến hóa Siêu tham số (Hyperparameter Evolution) kết hợp Kỹ thuật Tích lũy Gradient (Gradient Accumulation) nhằm vượt qua rào cản VRAM vật lý."
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
        "# Bổ sung torch kiểm tra hệ thống\n",
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
        "    print(\"Lỗi nghiêm trọng: Không phát hiện nhân tính toán CUDA. Quá trình phân tích sẽ bị gián đoạn và đình trệ.\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Kích hoạt Mô hình YOLOv11 Nano (YOLO11n) và Giải thuật Tiến hóa (Evolve)\n",
        "Áp dụng cấu hình phân rã khối lượng công việc nhằm bảo đảm an toàn VRAM (4GB):\n",
        "- **Kích thước vật lý (Batch Size)** = 8 (Mức độ dung nạp tối đa của VRAM 4GB).\n",
        "- **Hệ số giả lập Gradient (Accumulate)** = 8. Hệ thống tiến hành tích lũy ma trận sai số qua 8 chu kỳ trước khi tối ưu (8 x 8 = 64). Thuật toán này định tuyến mô hình hoạt động tương đương thiết lập Batch Size 64, nhằm ổn định Vector Gradient đối với ảnh thiếu sáng.\n",
        "- **Giải thuật Tiến hóa (Hyperparameter Evolve)**: Chạy 30 thế hệ (Generations) nhằm lai ghép tổ hợp siêu tham số tối ưu nhất cho môi trường thiếu sáng."
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
        "# Khởi động quá trình Tiến hóa Siêu tham số\n",
        "# LƯU Ý KỸ THUẬT: Quy trình yêu cầu thời gian điện toán liên tục. Tránh làm gián đoạn nguồn điện.\n",
        "results = model.train(\n",
        f"    data='d:/DAT301m/proposal/data/{yaml_name}',\n",
        "    epochs=50,\n",
        "    batch=8,               # Kích thước khung nạp vật lý\n",
        "    accumulate=8,          # Hệ số tích lũy (Mô phỏng kích thước lô 64)\n",
        "    imgsz=640,\n",
        "    device=0,\n",
        "    optimizer='AdamW',     # Giải thuật tối ưu chuyên biệt cho tập nhiễu hạt\n",
        "    evolve=30,             # Số lượng chu kỳ tiến hóa lai ghép gen\n",
        "    project='d:/DAT301m/proposal/models/runs',\n",
        f"    name='{title.lower().replace(\" \", \"_\")}_evolve'\n",
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
