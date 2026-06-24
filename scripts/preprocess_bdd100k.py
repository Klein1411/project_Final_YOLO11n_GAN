import os
import json
import shutil
import ijson
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = 'd:/DAT301m/proposal'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data/raw/bdd100k_night')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data/processed')

# Đường dẫn file JSON nguyên bản
JSON_TRAIN_PATH = os.path.join(RAW_DATA_DIR, 'labels/bdd100k_labels_images_train.json')
JSON_VAL_PATH = os.path.join(RAW_DATA_DIR, 'labels/bdd100k_labels_images_val.json')

# Thư mục ảnh nguyên bản
IMG_TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'images/train')
IMG_VAL_DIR = os.path.join(RAW_DATA_DIR, 'images/val')

# --- CẤU HÌNH LỚP (CLASSES) ---
# Tương ứng với file bdd100k.yaml
BDD_CLASSES = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9
}

# Hằng số chuẩn hóa tọa độ
IMG_WIDTH = 1280.0
IMG_HEIGHT = 720.0

def create_dirs(base_path):
    """Tạo cấu trúc thư mục YOLO"""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)

def safe_link_or_copy(src, dst):
    """
    Thử tạo Symlink (Shortcut) để tiết kiệm ổ cứng.
    Nếu Windows báo lỗi thiếu quyền (PermissionError), tự động chuyển sang Copy vật lý.
    """
    if os.path.exists(dst):
        return # Đã tồn tại, bỏ qua
    try:
        os.symlink(src, dst)
    except OSError:
        # Lỗi quyền Admin hoặc ổ đĩa không hỗ trợ -> Fallback copy
        shutil.copy2(src, dst)

def process_split(json_path, raw_img_dir, split_name):
    print(f"\n--- Bắt đầu xử lý tập: {split_name.upper()} ---")
    if not os.path.exists(json_path):
        print(f"Không tìm thấy file: {json_path}")
        return

    # Khởi tạo thư mục đích
    day_dir = os.path.join(PROCESSED_DIR, 'bdd_day')
    night_dir = os.path.join(PROCESSED_DIR, 'bdd_night')
    create_dirs(day_dir)
    create_dirs(night_dir)

    # Thống kê
    stats = {'daytime': 0, 'night': 0, 'dawn/dusk': 0, 'missing_img': 0, 'empty_labels': 0}
    
    # Do ijson không hỗ trợ lấy độ dài mảng (len), ta dùng con số ước lượng của BDD100k (70k train, 10k val)
    total_est = 70000 if split_name == 'train' else 10000
    
    # Mở file JSON dạng Stream (Chống tràn RAM)
    with open(json_path, 'r', encoding='utf-8') as f:
        # Phân tích từng item trong mảng root []
        items = ijson.items(f, 'item')
        
        for item in tqdm(items, total=total_est, desc=f"Đọc {split_name}"):
            img_name = item.get('name', '')
            attrs = item.get('attributes', {})
            timeofday = attrs.get('timeofday', '')

            # Lọc bỏ dawn/dusk
            if timeofday == 'dawn/dusk':
                stats['dawn/dusk'] += 1
                continue
            
            # Định tuyến thư mục
            if timeofday == 'daytime':
                target_base = day_dir
                stats['daytime'] += 1
            elif timeofday == 'night':
                target_base = night_dir
                stats['night'] += 1
            else:
                continue # Bỏ qua các nhãn không xác định

            # Kiểm tra ảnh gốc có tồn tại không
            src_img_path = os.path.join(raw_img_dir, img_name)
            if not os.path.exists(src_img_path):
                stats['missing_img'] += 1
                continue

            # Tiền xử lý Bounding Boxes
            yolo_labels = []
            labels = item.get('labels', [])
            for lbl in labels:
                category = lbl.get('category', '')
                lbl_attrs = lbl.get('attributes', {})
                
                # 1. Bỏ qua các Box bị che khuất (occluded) để giảm nhiễu
                if lbl_attrs.get('occluded', False):
                    continue
                
                # 2. Bỏ qua các lớp không thuộc 10 lớp giao thông
                if category not in BDD_CLASSES:
                    continue
                    
                class_id = BDD_CLASSES[category]
                box2d = lbl.get('box2d', {})
                if not box2d:
                    continue
                    
                # 3. Chuẩn hóa sang YOLO format
                x1, y1 = box2d['x1'], box2d['y1']
                x2, y2 = box2d['x2'], box2d['y2']
                
                # Cắt viền chống lỗi (clip)
                x1 = max(0.0, min(x1, IMG_WIDTH))
                x2 = max(0.0, min(x2, IMG_WIDTH))
                y1 = max(0.0, min(y1, IMG_HEIGHT))
                y2 = max(0.0, min(y2, IMG_HEIGHT))

                w = (x2 - x1)
                h = (y2 - y1)
                x_center = x1 + w / 2
                y_center = y1 + h / 2

                # Đưa về dải 0-1
                x_center /= IMG_WIDTH
                y_center /= IMG_HEIGHT
                w /= IMG_WIDTH
                h /= IMG_HEIGHT

                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # Nếu ảnh không có nhãn hợp lệ nào sau khi lọc, có thể bỏ qua hoặc giữ lại
            # Ở đây ta giữ lại (coi như ảnh nền - background image) vì nó tăng khả năng nhận diện False Positives
            if len(yolo_labels) == 0:
                stats['empty_labels'] += 1
            
            # Ghi ra file
            dst_img_path = os.path.join(target_base, split_name, 'images', img_name)
            dst_lbl_path = os.path.join(target_base, split_name, 'labels', img_name.replace('.jpg', '.txt'))
            
            # Ghi file TXT
            with open(dst_lbl_path, 'w', encoding='utf-8') as txt_f:
                txt_f.write('\n'.join(yolo_labels))
                
            # Tạo lối tắt hoặc Copy ảnh
            safe_link_or_copy(src_img_path, dst_img_path)

    print(f"\nThống kê {split_name}:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    print("=== BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ BDD100k ===")
    print("Ghi chú: Script đang sử dụng ijson để chống tràn RAM.")
    print("Sử dụng thuật toán Symlink Fallback: Nếu không có quyền Admin, tự động chuyển sang Copy ảnh.\n")
    
    process_split(JSON_TRAIN_PATH, IMG_TRAIN_DIR, 'train')
    process_split(JSON_VAL_PATH, IMG_VAL_DIR, 'val')
    
    print("\n=== HOÀN TẤT ===")
    print("Vui lòng cập nhật lại đường dẫn trong file bdd100k.yaml để trỏ vào bdd_day hoặc bdd_night.")
