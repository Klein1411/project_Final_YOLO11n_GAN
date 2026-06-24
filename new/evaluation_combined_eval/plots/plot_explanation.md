# Cach doc va danh gia cac bieu do

File nay giai thich cach nhin cac bieu do duoc tao tu `per_image_metrics.csv`.
Muc tieu la so sanh anh low-light ban dau voi anh da enhance cua model, deu doi chieu voi ground truth.

## 1. `mean_metrics_comparison.png`

Bieu do nay so sanh gia tri trung binh cua tung metric tren toan bo tap danh gia.

- Cot `Low vs GT`: chat luong anh dau vao low-light so voi ground truth.
- Cot `Enhanced vs GT`: chat luong anh model tao ra so voi ground truth.
- Voi PSNR va SSIM, cot Enhanced cao hon Low la tot.

Dung bieu do nay de ket luan tong quan model co cai thien anh hay khong.

## 2. PSNR

PSNR la metric `cang cao cang tot`.

Gia tri trung binh Low vs GT: `9.2119`.
Gia tri trung binh Enhanced vs GT: `17.2767`.
Muc cai thien trung binh: `8.0649`.
So anh cai thien: `178/199`.
So anh xau hon: `21/199`.

### `psnr_per_image.png`

Bieu do nay cho thay metric tren tung anh.

- Duong `Low vs GT` la diem cua anh low-light ban dau.
- Duong `Enhanced vs GT` la diem cua anh sau khi qua model.
- Voi PSNR, cang cao cang tot.

Cach danh gia:

- Neu duong Enhanced tot hon Low tren hau het anh, model cai thien on dinh.
- Neu hai duong gan nhau, model cai thien it.
- Neu Enhanced te hon Low o mot so anh, can xem lai cac anh do vi model co the lam sai mau, mat chi tiet, hoac lam sang qua muc.

### `psnr_change_per_image.png`

Bieu do nay ve muc `gain` cua tung anh.

- Gia tri duong: duong nghia la metric tang, anh enhanced tot hon anh low.
- Gia tri gan 0: model gan nhu khong cai thien.
- Gia tri am: model lam anh te hon theo metric nay.

Anh cai thien manh nhat: `lol_128` voi muc `gain` = `19.3335`.
Anh cai thien kem nhat: `sice_Dataset_Part1_133` voi muc `gain` = `-35.5760`.

## 3. SSIM

SSIM la metric `cang cao cang tot`.

Gia tri trung binh Low vs GT: `0.3411`.
Gia tri trung binh Enhanced vs GT: `0.7694`.
Muc cai thien trung binh: `0.4283`.
So anh cai thien: `182/199`.
So anh xau hon: `17/199`.

### `ssim_per_image.png`

Bieu do nay cho thay metric tren tung anh.

- Duong `Low vs GT` la diem cua anh low-light ban dau.
- Duong `Enhanced vs GT` la diem cua anh sau khi qua model.
- Voi SSIM, cang cao cang tot.

Cach danh gia:

- Neu duong Enhanced tot hon Low tren hau het anh, model cai thien on dinh.
- Neu hai duong gan nhau, model cai thien it.
- Neu Enhanced te hon Low o mot so anh, can xem lai cac anh do vi model co the lam sai mau, mat chi tiet, hoac lam sang qua muc.

### `ssim_change_per_image.png`

Bieu do nay ve muc `gain` cua tung anh.

- Gia tri duong: duong nghia la metric tang, anh enhanced tot hon anh low.
- Gia tri gan 0: model gan nhu khong cai thien.
- Gia tri am: model lam anh te hon theo metric nay.

Anh cai thien manh nhat: `lol_676` voi muc `gain` = `0.7719`.
Anh cai thien kem nhat: `sice_Dataset_Part1_133` voi muc `gain` = `-0.7316`.

## 4. Cach ket luan ngan gon

Khi viet bao cao, nen ket luan theo thu tu:

1. Nhin `mean_metrics_comparison.png` de noi model co cai thien tong the khong.
2. Nhin cac bieu do `*_per_image.png` de xem cai thien co dong deu tren tung anh khong.
3. Nhin cac bieu do `*_change_per_image.png` de tim anh tot nhat va anh yeu nhat.
4. Neu PSNR tang nhung SSIM khong tang nhieu, model co the dung pixel hon nhung chua giu cau truc tot.
5. Neu SSIM tang nhung PSNR thap, anh co the giu cau truc nhung con lech do sang hoac mau.

Ket luan tot nen dua tren nhieu metric cung luc, khong dua vao mot bieu do duy nhat.
