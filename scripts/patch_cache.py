import json
import glob

for nb_path in glob.glob(r'd:\DAT301m\proposal\notebooks\0[23]_*.ipynb'):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Check if cache is already present
            has_cache = any('cache=' in l for l in source)
            if not has_cache:
                for i, line in enumerate(source):
                    if 'batch=8' in line:
                        source.insert(i+1, "    cache='disk',          # Bộ đệm ổ cứng giảm tải System RAM\n")
                        modified = True
                        break
    
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Patched {nb_path}")
    else:
        print(f"Already patched or not found in {nb_path}")
