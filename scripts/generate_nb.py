import os
import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Giai đoạn 1: Chuẩn bị và Chuyển đổi Dữ liệu (Data Conversion)\n",
    "Notebook này thực hiện các tác vụ:\n",
    "1. Convert label của tập **ExDark** sang chuẩn YOLO (kèm xử lý tọa độ tràn viền).\n",
    "2. Áp dụng tiền xử lý **CLAHE + Median Filter** để tạo ra tập baseline truyền thống.\n",
    "\n",
    "**⚠️ YÊU CẦU TRƯỚC KHI CHẠY CHƯƠNG TRÌNH NÀY:**\n",
    "Ngài hãy tải và chép dữ liệu raw của các tổ chức vào đúng cấu trúc sau:\n",
    "- `data/raw/exdark/images/`\n",
    "- `data/raw/exdark/labels/` (Chứa các file txt gốc)\n",
    "- `data/raw/bdd100k_night/images/`\n",
    "- `data/raw/bdd100k_night/labels/bdd100k_labels_images_train.json`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "from tqdm import tqdm\n",
    "import shutil\n",
    "import random\n",
    "\n",
    "# Fix random seed để chia split luôn cố định\n",
    "random.seed(42)\n",
    "\n",
    "BASE_DIR = Path(os.getcwd()).parent\n",
    "RAW_DIR = BASE_DIR / 'data' / 'raw'\n",
    "PROCESSED_DIR = BASE_DIR / 'data' / 'processed'\n",
    "CLAHE_DIR = BASE_DIR / 'data' / 'clahe_baseline'\n",
    "\n",
    "# Khởi tạo các thư mục đầu ra\n",
    "for split in ['train', 'val']:\n",
    "    (PROCESSED_DIR / split / 'images').mkdir(parents=True, exist_ok=True)\n",
    "    (PROCESSED_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)\n",
    "    (CLAHE_DIR / split / 'images').mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "print(\"✅ Đã khởi tạo cấu trúc thư mục đầu ra!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Xử lý tập ExDark (12 Classes)\n",
    "- **Rủi ro đã xử lý**: Tọa độ của ExDark là 1-indexed. Có chứa header rác `% bbGt`. Có thể bị tràn viền ảnh.\n",
    "- **Split**: 80% Train - 20% Val."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "EXDARK_CLASSES = ['bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cup', 'dog', 'motorbike', 'people', 'table']\n",
    "EXDARK_CLASS_MAP = {name: i for i, name in enumerate(EXDARK_CLASSES)}\n",
    "\n",
    "def convert_exdark():\n",
    "    raw_img_dir = RAW_DIR / 'exdark' / 'images'\n",
    "    raw_label_dir = RAW_DIR / 'exdark' / 'labels'\n",
    "    \n",
    "    img_paths = list(raw_img_dir.rglob('*.*'))\n",
    "    if not img_paths:\n",
    "        print(\"⚠️ Chưa có data ExDark. Ngài hãy copy data vào thư mục raw nhé!\")\n",
    "        return\n",
    "        \n",
    "    print(f\"Bắt đầu convert {len(img_paths)} ảnh ExDark...\")\n",
    "    \n",
    "    random.shuffle(img_paths)\n",
    "    split_idx = int(len(img_paths) * 0.8)\n",
    "    train_paths = img_paths[:split_idx]\n",
    "    val_paths = img_paths[split_idx:]\n",
    "    \n",
    "    def process_split(paths, split_name):\n",
    "        count = 0\n",
    "        for img_path in tqdm(paths, desc=f\"ExDark {split_name}\"):\n",
    "            img = cv2.imread(str(img_path))\n",
    "            if img is None: continue\n",
    "            h_img, w_img, _ = img.shape\n",
    "            \n",
    "            rel_path = img_path.relative_to(raw_img_dir)\n",
    "            label_path = raw_label_dir / rel_path.parent / (img_path.name + '.txt')\n",
    "            if not label_path.exists():\n",
    "                label_path = raw_label_dir / rel_path.parent / (img_path.stem + '.txt')\n",
    "                if not label_path.exists(): continue\n",
    "                \n",
    "            yolo_labels = []\n",
    "            with open(label_path, 'r', encoding='utf-8') as f:\n",
    "                for line in f:\n",
    "                    line = line.strip()\n",
    "                    if not line or line.startswith('%'): continue\n",
    "                    \n",
    "                    parts = line.split()\n",
    "                    if len(parts) < 5: continue\n",
    "                    \n",
    "                    class_name = parts[0].lower()\n",
    "                    if class_name not in EXDARK_CLASS_MAP: continue\n",
    "                    class_id = EXDARK_CLASS_MAP[class_name]\n",
    "                    \n",
    "                    l = max(0.0, float(parts[1]) - 1.0)\n",
    "                    t = max(0.0, float(parts[2]) - 1.0)\n",
    "                    w = float(parts[3])\n",
    "                    h = float(parts[4])\n",
    "                    \n",
    "                    x_center = np.clip((l + w / 2) / w_img, 0, 1)\n",
    "                    y_center = np.clip((t + h / 2) / h_img, 0, 1)\n",
    "                    w_norm = np.clip(w / w_img, 0, 1)\n",
    "                    h_norm = np.clip(h / h_img, 0, 1)\n",
    "                    \n",
    "                    if w_norm == 0 or h_norm == 0: continue\n",
    "                    yolo_labels.append(f\"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\\n\")\n",
    "            \n",
    "            if yolo_labels:\n",
    "                out_img_path = PROCESSED_DIR / split_name / 'images' / img_path.name\n",
    "                out_label_path = PROCESSED_DIR / split_name / 'labels' / (img_path.stem + '.txt')\n",
    "                shutil.copy(img_path, out_img_path)\n",
    "                with open(out_label_path, 'w', encoding='utf-8') as f:\n",
    "                    f.writelines(yolo_labels)\n",
    "                count += 1\n",
    "        print(f\"✅ Đã xử lý {count} ảnh cho tập {split_name}.\")\n",
    "\n",
    "    process_split(train_paths, 'train')\n",
    "    process_split(val_paths, 'val')\n",
    "\n",
    "# Bỏ comment dòng dưới để chạy\n",
    "# convert_exdark()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Tiền xử lý Baseline Truyền thống (CLAHE + Median Filter)\n",
    "Tạo ra một tập dữ liệu sáng tự nhiên bằng thuật toán truyền thống để làm cột mốc (baseline) so sánh xem GAN có thực sự vượt trội hay không."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_clahe(img_path, out_path):\n",
    "    img = cv2.imread(str(img_path))\n",
    "    if img is None: return\n",
    "    \n",
    "    # Khử nhiễu ISO ban đêm bằng Median Blur\n",
    "    img = cv2.medianBlur(img, 3)\n",
    "    \n",
    "    # Áp dụng CLAHE trên kênh L (Lightness)\n",
    "    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)\n",
    "    l, a, b = cv2.split(lab)\n",
    "    \n",
    "    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\n",
    "    cl = clahe.apply(l)\n",
    "    \n",
    "    limg = cv2.merge((cl, a, b))\n",
    "    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)\n",
    "    cv2.imwrite(str(out_path), final)\n",
    "\n",
    "def create_clahe_dataset():\n",
    "    print(\"Bắt đầu sinh ảnh CLAHE Baseline...\")\n",
    "    for split in ['train', 'val']:\n",
    "        in_dir = PROCESSED_DIR / split / 'images'\n",
    "        out_dir = CLAHE_DIR / split / 'images'\n",
    "        \n",
    "        img_paths = list(in_dir.glob('*.*'))\n",
    "        for img_path in tqdm(img_paths, desc=f\"CLAHE {split}\"):\n",
    "            out_path = out_dir / img_path.name\n",
    "            preprocess_clahe(img_path, out_path)\n",
    "            \n",
    "            in_label = PROCESSED_DIR / split / 'labels' / (img_path.stem + '.txt')\n",
    "            out_label = CLAHE_DIR / split / 'labels' / (img_path.stem + '.txt')\n",
    "            (CLAHE_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)\n",
    "            if in_label.exists():\n",
    "                shutil.copy(in_label, out_label)\n",
    "    print(\"✅ Hoàn tất CLAHE!\")\n",
    "\n",
    "# Bỏ comment dòng dưới để chạy\n",
    "# create_clahe_dataset()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "project final DAT301m",
   "language": "python",
   "name": "project_final_dat301m"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open(r'd:\DAT301m\proposal\notebooks\01_data_conversion.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print("Notebook 01 created successfully.")
