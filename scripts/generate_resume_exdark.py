import json

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Giai đoạn 3 (Tiếp nối): Resume Training ExDark +30 Epochs\n",
    "Notebook này tiếp tục quá trình huấn luyện từ **Epoch 50** (file `last.pt`).\n",
    "- Model vẫn đang trong xu hướng cải thiện (chưa bão hòa) → quyết định vắt kiệt tiềm năng.\n",
    "- Tổng cộng sau khi hoàn tất: **80 Epochs** (50 cũ + 30 mới).\n",
    "- Tích hợp bộ đếm thời gian tổng để theo dõi chi phí tính toán."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Kiểm tra GPU & VRAM\n",
    "import torch\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    print(f\"VRAM: {vram:.2f} GB\")\n",
    "else:\n",
    "    print(\"CẢNH BÁO: Không phát hiện CUDA GPU!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Lưu Mốc Lịch Sử (Checkpoint Milestone - Epoch 50)\n",
    "Trước khi Resume ghi đè, sao lưu toàn bộ trọng số và bảng số liệu của Pha 1 (50 Epochs) thành bản đánh dấu mốc.\n",
    "- `last.pt` → `last_epoch50.pt` (trọng số tại thời điểm dừng)\n",
    "- `best.pt` → `best_epoch50.pt` (trọng số tốt nhất trong 50 epoch đầu)\n",
    "- `results.csv` → `results_epoch50.csv` (toàn bộ log huấn luyện Pha 1)\n",
    "- Các biểu đồ → lưu vào thư mục con `milestone_epoch50/`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil, os\n",
    "\n",
    "run_dir = 'd:/DAT301m/proposal/models/runs/exdark_train-2'\n",
    "milestone_dir = os.path.join(run_dir, 'milestone_epoch50')\n",
    "os.makedirs(milestone_dir, exist_ok=True)\n",
    "\n",
    "# Sao lưu trọng số\n",
    "shutil.copy2(f'{run_dir}/weights/last.pt', f'{run_dir}/weights/last_epoch50.pt')\n",
    "shutil.copy2(f'{run_dir}/weights/best.pt', f'{run_dir}/weights/best_epoch50.pt')\n",
    "\n",
    "# Sao lưu bảng số liệu\n",
    "shutil.copy2(f'{run_dir}/results.csv', f'{milestone_dir}/results_epoch50.csv')\n",
    "\n",
    "# Sao lưu biểu đồ & ma trận\n",
    "for fname in ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png',\n",
    "              'BoxPR_curve.png', 'BoxF1_curve.png', 'BoxP_curve.png', 'BoxR_curve.png']:\n",
    "    src = os.path.join(run_dir, fname)\n",
    "    if os.path.exists(src):\n",
    "        shutil.copy2(src, os.path.join(milestone_dir, fname))\n",
    "\n",
    "print(f'Checkpoint Milestone Epoch 50 saved to: {milestone_dir}')\n",
    "print(f'Weights backup: last_epoch50.pt, best_epoch50.pt')\n",
    "print(f'Files saved: {os.listdir(milestone_dir)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Resume Training từ last.pt\n",
    "- Nạp trọng số từ Epoch 50 (`last.pt`) thay vì `yolo11n.pt` (trọng số COCO gốc).\n",
    "- Tổng epochs = **80** (hệ thống tự biết đã train 50 rồi, sẽ chỉ chạy thêm 30 epoch nữa).\n",
    "- Giữ nguyên toàn bộ siêu tham số để đảm bảo tính nhất quán thí nghiệm.\n",
    "- Kèm bộ đếm thời gian tổng (phút)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "from ultralytics import YOLO\n",
    "\n",
    "# === BỘ ĐẾM THỜI GIAN TỔNG ===\n",
    "start_time = time.time()\n",
    "\n",
    "# Nạp trọng số từ lần train trước (Epoch 50)\n",
    "model = YOLO('d:/DAT301m/proposal/models/runs/exdark_train-2/weights/last.pt')\n",
    "\n",
    "# Resume Training thêm 30 Epochs (tổng = 80)\n",
    "results = model.train(\n",
    "    data='d:/DAT301m/proposal/data/exdark.yaml',\n",
    "    epochs=80,              # Tổng epochs mong muốn (hệ thống tự tính: 80 - 50 = 30 epochs còn lại)\n",
    "    patience=15,            # Dừng sớm nếu sau 15 epoch mAP không cải thiện\n",
    "    batch=8,                # Giữ nguyên Batch Size\n",
    "    cache='disk',           # Bộ đệm ổ cứng giảm tải System RAM\n",
    "    imgsz=640,\n",
    "    device=0,\n",
    "    optimizer='AdamW',\n",
    "    mosaic=0.5,\n",
    "    close_mosaic=10,        # Tắt Mosaic ở 10 epoch cuối (epoch 70-80)\n",
    "    resume=True,            # BẮT BUỘC: Kích hoạt cơ chế Resume\n",
    "    project='d:/DAT301m/proposal/models/runs',\n",
    "    name='exdark_train-2',\n",
    "    exist_ok=True           # Cho phép ghi tiếp vào thư mục cũ\n",
    ")\n",
    "\n",
    "# === KẾT THÚC - BÁO CÁO THỜI GIAN ===\n",
    "elapsed = time.time() - start_time\n",
    "print(f\"\\n{'='*50}\")\n",
    "print(f\"HOÀN TẤT RESUME TRAINING\")\n",
    "print(f\"Thời gian pha Resume: {elapsed/60:.1f} phút ({elapsed:.0f} giây)\")\n",
    "print(f\"Tổng thời gian tích lũy (cả 2 pha): ~{(7230 + elapsed)/60:.1f} phút\")\n",
    "print(f\"{'='*50}\")"
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

nb_path = 'd:/DAT301m/proposal/notebooks/02b_resume_exdark.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Created: {nb_path}")
