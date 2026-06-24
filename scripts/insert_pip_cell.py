import json

nb_path = r'd:\DAT301m\proposal\notebooks\01_data_conversion.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Kiểm tra xem đã có cell chứa pip install chưa
has_install = any('!{sys.executable} -m pip install' in "".join(cell.get('source', [])) for cell in nb['cells'])

if not has_install:
    install_cell = {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# Cài đặt các thư viện lõi trước khi chạy toàn bộ Pipeline\n",
      "import sys\n",
      "!{sys.executable} -m pip install opencv-python \"numpy<2.0.0\" tqdm ijson"
     ]
    }
    
    # Chèn vào vị trí số 1 (ngay dưới ô Markdown Header đầu tiên)
    nb['cells'].insert(1, install_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("✅ Đã chèn thành công lệnh tải thư viện vào Notebook!")
else:
    print("⚠️ Lệnh tải thư viện đã tồn tại trong Notebook.")
