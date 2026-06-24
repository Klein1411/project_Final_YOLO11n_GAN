import json
import os

nb_path = r'd:\DAT301m\proposal\notebooks\01_data_conversion.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Kiểm tra xem đã có cell BDD chưa để tránh append nhiều lần
has_bdd = any('3. Xử lý tập BDD100k' in str(cell.get('source', '')) for cell in nb['cells'])

if not has_bdd:
    markdown_cell = {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## 3. Xử lý tập BDD100k (10 Classes) - Chế độ Stream JSON\n",
      "- **Rủi ro đã xử lý**: File JSON cực kỳ khổng lồ (1.4GB), nếu load bằng `json.load` cơ bản sẽ làm Crash RAM (OOM) của máy. Bắt buộc dùng `ijson` để đọc theo dạng dòng chảy (stream).\n",
      "- **Lọc nhiễu**: Chỉ lấy các ảnh có cờ `timeofday == \"night\"`. Bỏ qua toàn bộ các box rác có cờ `occluded == True` (bị che lấp quá nửa) để model tập trung học viền nét vật thể thật."
     ]
    }

    code_cell = {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "import sys\n",
      "import numpy as np\n",
      "try:\n",
      "    import ijson\n",
      "except ImportError:\n",
      "    print(\"Đang cài đặt thư viện ijson siêu nhẹ để chống Crash RAM...\")\n",
      "    !{sys.executable} -m pip install ijson\n",
      "    import ijson\n",
      "\n",
      "BDD_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']\n",
      "BDD_CLASS_MAP = {name: i for i, name in enumerate(BDD_CLASSES)}\n",
      "\n",
      "def convert_bdd100k_stream():\n",
      "    bdd_img_dir = RAW_DIR / 'bdd100k_night' / 'images'\n",
      "    bdd_label_dir = RAW_DIR / 'bdd100k_night' / 'labels'\n",
      "    \n",
      "    for split in ['train', 'val']:\n",
      "        json_file = bdd_label_dir / f\"bdd100k_labels_images_{split}.json\"\n",
      "        if not json_file.exists():\n",
      "            print(f\"⚠️ Không tìm thấy file {json_file.name}. Bỏ qua!\")\n",
      "            continue\n",
      "            \n",
      "        out_img_split = PROCESSED_DIR / split / 'bdd_images'\n",
      "        out_lbl_split = PROCESSED_DIR / split / 'bdd_labels'\n",
      "        out_img_split.mkdir(parents=True, exist_ok=True)\n",
      "        out_lbl_split.mkdir(parents=True, exist_ok=True)\n",
      "        \n",
      "        print(f\"\\nBắt đầu Stream JSON {split}...\")\n",
      "        count_night = 0\n",
      "        count_boxes = 0\n",
      "        \n",
      "        with open(json_file, 'r', encoding='utf-8') as f:\n",
      "            # Dùng ijson đọc từng item trong Array thay vì nạp toàn bộ vào RAM\n",
      "            for item in tqdm(ijson.items(f, 'item'), desc=f\"BDD100k {split}\"):\n",
      "                attrs = item.get('attributes', {})\n",
      "                if attrs.get('timeofday') != 'night':\n",
      "                    continue  # Bỏ qua ảnh ngày\n",
      "                    \n",
      "                img_name = item.get('name')\n",
      "                if not img_name: continue\n",
      "                \n",
      "                # Tìm file ảnh gốc\n",
      "                img_src_path = bdd_img_dir / split / img_name\n",
      "                if not img_src_path.exists(): continue\n",
      "                \n",
      "                # Resolution gốc của BDD100k mặc định là 1280x720\n",
      "                w_img, h_img = 1280.0, 720.0\n",
      "                \n",
      "                yolo_lines = []\n",
      "                for label in item.get('labels', []):\n",
      "                    cat = label.get('category')\n",
      "                    if cat not in BDD_CLASS_MAP: continue\n",
      "                    \n",
      "                    # Lọc bỏ rác bị che khuất\n",
      "                    if label.get('attributes', {}).get('occluded') == True:\n",
      "                        continue\n",
      "                        \n",
      "                    box2d = label.get('box2d')\n",
      "                    if not box2d: continue\n",
      "                    \n",
      "                    x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']\n",
      "                    w = x2 - x1\n",
      "                    h = y2 - y1\n",
      "                    x_center = np.clip((x1 + w / 2) / w_img, 0, 1)\n",
      "                    y_center = np.clip((y1 + h / 2) / h_img, 0, 1)\n",
      "                    w_norm = np.clip(w / w_img, 0, 1)\n",
      "                    h_norm = np.clip(h / h_img, 0, 1)\n",
      "                    \n",
      "                    if w_norm == 0 or h_norm == 0: continue\n",
      "                    class_id = BDD_CLASS_MAP[cat]\n",
      "                    yolo_lines.append(f\"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\\n\")\n",
      "                    count_boxes += 1\n",
      "                \n",
      "                # Nếu ảnh đêm đó có chứa box hợp lệ thì mới copy và lưu nhãn\n",
      "                if yolo_lines:\n",
      "                    out_img_path = out_img_split / img_name\n",
      "                    out_lbl_path = out_lbl_split / (img_src_path.stem + '.txt')\n",
      "                    \n",
      "                    shutil.copy(img_src_path, out_img_path)\n",
      "                    with open(out_lbl_path, 'w', encoding='utf-8') as fl:\n",
      "                        fl.writelines(yolo_lines)\n",
      "                    count_night += 1\n",
      "                    \n",
      "        print(f\"✅ {split}: Rút trích thành công {count_night} ảnh ĐÊM và {count_boxes} nhãn siêu sạch!\")\n",
      "\n",
      "# Bỏ comment dòng dưới để chạy\n",
      "# convert_bdd100k_stream()"
     ]
    }
    
    nb['cells'].append(markdown_cell)
    nb['cells'].append(code_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Đã chèn thành công thuật toán BDD100k vào Notebook 01!")
else:
    print("Notebook đã có đoạn code BDD100k rồi.")
