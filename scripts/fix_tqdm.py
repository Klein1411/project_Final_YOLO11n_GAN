import json

nb_path = r'd:\DAT301m\proposal\notebooks\01_data_conversion.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'from tqdm.notebook import tqdm' in line:
                source[i] = line.replace('from tqdm.notebook import tqdm', 'from tqdm import tqdm')
                changed = True

if changed:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook đã được fix lỗi tqdm.")
else:
    print("Không tìm thấy import bị lỗi.")
