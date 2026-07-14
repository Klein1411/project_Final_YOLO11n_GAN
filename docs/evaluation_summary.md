# Tổng Hợp Đánh Giá YOLO

Tài liệu này tổng hợp kết quả sau khi hoàn tất base train BDD day, fine-tune BDD night và validation chéo. Số liệu BDD được lấy từ [bdd_cross_domain_metrics.csv](../models/runs/bdd_evaluation/bdd_cross_domain_metrics.csv). Số liệu ExDark được lấy từ [results.csv](../models/runs/exdark_train-2/results.csv).

## 1. BDD Cross-Domain

| Model | Tập validation | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Day model | Day | 0,674 | 0,399 | 0,442 | 0,248 |
| Day model | Night | 0,564 | 0,329 | 0,333 | 0,181 |
| Night fine-tune | Night | 0,635 | 0,408 | 0,427 | 0,226 |
| Night fine-tune | Day | 0,598 | 0,327 | 0,346 | 0,191 |

### Diễn giải

- Trên miền night, night fine-tune tăng `mAP50-95` từ `0,18088` lên `0,22647`, tương đương khoảng `+0,04559`.
- Trên miền day, night fine-tune giảm `mAP50-95` từ `0,24823` xuống `0,19056`, tương đương khoảng `-0,05768`.
- Fine-tune theo miền đã cải thiện khả năng thích nghi với dữ liệu tối nhưng làm giảm khả năng duy trì hiệu năng trên miền ban ngày.
- Vì vậy, không nên gọi night fine-tune là model tốt hơn tuyệt đối. Nó là model chuyên biệt cho night; day model vẫn là checkpoint phù hợp hơn nếu mục tiêu bao phủ cả hai miền.

## 2. ExDark

Kết quả cuối của run `exdark_train-2` sau 80 epoch resume:

| Metric | Giá trị |
| --- | ---: |
| Precision | 0,692 |
| Recall | 0,575 |
| mAP50 | 0,636 |
| mAP50-95 | 0,393 |

ExDark có 12 lớp, trong khi BDD có 10 lớp giao thông và được tách thành hai miền thời gian. Các giá trị mAP của hai dataset không được gộp thành một thứ hạng tuyệt đối vì khác class ontology, phân phối ảnh, kích thước vật thể và protocol dữ liệu. ExDark được dùng làm baseline tham chiếu độc lập cho pipeline YOLO low-light.

## 3. Nhận Định Theo Lớp

- `car` là lớp ổn định và mạnh nhất trong các log validation BDD.
- `train` có rất ít mẫu, nên precision có thể nhìn cao nhưng recall và mAP gần như bằng 0; không được diễn giải là model học tốt lớp này.
- `rider` yếu hơn do số mẫu ít và dễ nhầm với `pedestrian`.
- `motorcycle`, `bicycle`, `traffic light` và `traffic sign` nhạy hơn với thay đổi miền sáng/tối và cần đọc cùng confusion matrix, không chỉ nhìn mAP tổng.
- Các quan sát này là mô tả chẩn đoán, không thay thế phân tích per-class đầy đủ trong báo cáo cuối.

## 4. Enhancement Legacy

Audit tại [enhancement_artifact_audit.csv](enhancement_artifact_audit.csv) đã tìm thấy các nhóm CLAHE, Gamma, MSRCR, Median Filter, GAN và artifact tổng hợp cũ. Hiện các nhóm này đều có trạng thái `manual_review_required` vì chưa đủ bằng chứng về:

- tập ảnh và nhãn tương ứng;
- pipeline tạo ảnh và tham số enhancement;
- checkpoint detector và protocol đánh giá;
- khả năng tái lập kết quả trên cùng split.

Do đó, các artifact này bị loại khỏi bảng định lượng chính. Có thể dùng chúng ở phụ lục minh họa định tính nếu cần, nhưng không dùng để khẳng định enhancement nào cải thiện detector.

## 5. Hạn Chế Và Threat To Validity

- BDD day và BDD night có phân phối khác nhau; chênh lệch mAP phản ánh cả domain shift, không chỉ chất lượng fine-tune.
- Bảng cross-domain dùng validation split, chưa phải một test set độc lập hoàn toàn.
- Hai model dùng cùng kiến trúc YOLO11n và cấu hình chính, nhưng night model khởi tạo từ day checkpoint nên đây là warm-start fine-tuning, không phải train độc lập từ đầu.
- ExDark và BDD không cùng ontology, nên không được so sánh mAP trực tiếp.
- Các artifact GAN/enhancement cũ chưa đủ provenance để đưa vào kết luận khoa học.

## 6. Kết Luận Và Bước Tiếp Theo

- Pipeline BDD sau khi sửa alias đã hoàn tất và có kết quả cross-domain hợp lệ.
- Day model là checkpoint cân bằng hơn cho miền day; night fine-tune là checkpoint chuyên biệt cho miền night.
- Không train thêm trong phạm vi hiện tại.
- Bước tiếp theo là đưa các bảng và kết luận này vào báo cáo, kèm confusion matrix và một số mẫu inference tiêu biểu từ `models/runs/bdd_evaluation/`.
