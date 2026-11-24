# Pipeline thực hiện Đề tài 2 — Phân đoạn ảnh y tế

## 1. Phạm vi và yêu cầu

**Bài toán:** phân đoạn polyp trong ảnh nội soi. Đầu vào là ảnh RGB; đầu ra là mask nhị phân cùng kích thước ảnh gốc, với `1 = polyp`, `0 = nền`.

**Dữ liệu:** Kvasir-SEG để train/validation/test nội bộ; CVC-ClinicDB để đánh giá bên ngoài.

Kế hoạch đối chiếu với Đề tài 2

| Mức ưu tiên | Nội dung | Căn cứ |
| --- | --- | --- |
| Tối thiểu, làm trước | Tự cài U-Net từ đầu; so sánh U-Net dùng pretrained encoder; báo cáo Dice và IoU | Yêu cầu tối thiểu của Đề tài 2 |
| Mở rộng theo đề, nhóm dự kiến làm đủ ba hướng | SegFormer-B0; so sánh Dice/Focal/Tversky loss; SAM zero-shot | Các hướng mở rộng cô liệt kê; chọn SegFormer là đủ cho mục SegFormer hoặc TransUNet |
| Bổ sung của nhóm | Đánh giá trên ClinicDB; ResNet34 khởi tạo ngẫu nhiên; nhiều seed; phân tích theo kích thước polyp | Đề xuất để phân tích sâu hơn, không phải yêu cầu bắt buộc trong PDF |
| Sản phẩm và quá trình | Proposal, milestone, báo cáo ACL, repo, log chia sẻ, mã tái tạo kết quả, slide và vấn đáp | Quy định chung áp dụng cho Đề tài 2 |

Đề nêu phần mở rộng cần thiết để đạt mức giỏi nhưng không quy định số lượt train hay ngưỡng Dice bảo đảm điểm số. Ưu tiên cài đặt đúng, thí nghiệm có kiểm soát và giải thích được kết quả.


## 2. Bảng thí nghiệm dự kiến

| Mã | Mô hình | Khởi tạo encoder | Loss | Câu hỏi cần trả lời |
| --- | --- | --- | --- | --- |
| M1 | U-Net tự cài | Ngẫu nhiên | BCE + Dice | Baseline tự cài đạt kết quả như thế nào? |
| M2 | U-Net–ResNet34 | Pretrained ImageNet | BCE + Dice | Cấu hình pretrained U-Net khác baseline ra sao? |
| M3 | SegFormer-B0 | MiT-B0 pretrained ImageNet | BCE + Dice | SegFormer so với các cấu hình CNN đã thử như thế nào? |
| A1 | U-Net–ResNet34 | Ngẫu nhiên | BCE + Dice | Pretraining có tác dụng gì khi giữ nguyên kiến trúc? |
| L1 | U-Net–ResNet34 | Pretrained ImageNet | Dice | Ảnh hưởng của Dice loss |
| L2 | U-Net–ResNet34 | Pretrained ImageNet | Binary Focal | Ảnh hưởng của Focal loss |
| L3 | U-Net–ResNet34 | Pretrained ImageNet | Tversky | Ảnh hưởng của trọng số phạt false positive/false negative |
| S1 | SAM ViT-B, đóng băng | Checkpoint SAM gốc | Không train | Zero-shot với một điểm foreground lấy từ nhãn |
| S2 | SAM ViT-B, đóng băng | Cùng checkpoint S1 | Không train | Zero-shot với box lấy từ nhãn |

M1 và M2 phục vụ yêu cầu tối thiểu. M3, L1–L3 và S1–S2 phục vụ ba hướng mở rộng. A1 là ablation bổ sung của nhóm.

Có **7 cấu hình train** nếu làm cả A1; ba seed cho toàn bộ tương ứng **21 lượt train**, chưa tính chạy thử/tinh chỉnh. SAM chỉ chạy suy luận. Trước tiên chạy một seed để kiểm tra và ước lượng thời gian; sau khi khóa cấu hình, mục tiêu là lặp ba seed cho các so sánh chính, mở rộng sang các cấu hình còn lại khi đủ ngân sách. Công bố rõ số lượt thực tế và các cấu hình chưa lặp.

## 3. Thứ tự thực hiện

### Bước 1 — Chốt câu hỏi nghiên cứu, phân công và môi trường

Bài tập tập trung vào bốn câu hỏi:

1. U-Net tự cài và U-Net dùng pretrained encoder khác nhau thế nào? A1 giúp phân biệt ảnh hưởng của kiến trúc với ảnh hưởng của pretraining.
2. SegFormer-B0 có ưu/nhược điểm gì so với các cấu hình CNN được thử nghiệm?
3. Dice, Focal và Tversky ảnh hưởng thế nào đến mức chồng lấp, bỏ sót và dự đoán thừa?
4. Các mô hình giữ được hiệu năng ra sao trên ClinicDB, và SAM không fine-tune hoạt động thế nào với từng loại prompt?

Chia công việc cho ba thành viên, ghi người phụ trách và người đọc chéo từng phần. Có thể phân công: (1) dữ liệu và U-Net; (2) pretrained U-Net và SegFormer; (3) loss, SAM và tổng hợp đánh giá. Cả nhóm dùng chung split, metric và quy tắc chọn checkpoint.

### Bước 2 — Kiểm tra dữ liệu và cố định split

Dùng notebook [01_download_data.ipynb](notebooks/01_download_data.ipynb) để chuẩn bị dữ liệu. Giữ nguồn, checksum và revision như [README](README.md), không đổi mirror hoặc sửa dữ liệu nguồn khi chưa xác minh.

Thực hiện:

- Kiểm tra đủ 1.000 cặp Kvasir và 612 cặp ClinicDB; ghép ảnh–mask theo tên, không chỉ dựa vào thứ tự đọc thư mục.
- Kiểm tra khả năng đọc file, kích thước khớp nhau và hiển thị ảnh, mask, overlay.
- Thống kê độ phân giải, tỷ lệ diện tích polyp và số mask rỗng.
- Tìm ảnh trùng/gần trùng trong Kvasir trước khi chia; kiểm tra thêm giữa Kvasir và ClinicDB. Ghi phương pháp, ngưỡng phát hiện và kết quả kiểm tra thủ công.
- Nếu có ảnh liên quan, đưa chúng vào cùng nhóm khi chia. Nếu có trùng giữa train và ClinicDB, xử lý theo quy tắc xác định trước khi train và ghi lại danh sách, số lượng đánh giá thực tế.

**Split dự kiến:** Kvasir train/val/test = **70/15/15**, tương ứng **700/150/150** nếu không cần điều chỉnh theo nhóm. Seed chia tập là **42**. Nếu nhóm ảnh liên quan khiến không đạt đúng số lượng, ưu tiên không tách nhóm và công bố số lượng cuối.

Lưu ID cố định vào `splits/kvasir_train.txt`, `kvasir_val.txt`, `kvasir_test.txt` và `clinicdb_external.txt`. Kiểm tra các tập Kvasir không giao nhau và bao phủ danh sách dự kiến. Mọi mô hình dùng cùng split.

ClinicDB chỉ dùng đánh giá ngoài: không dùng để chọn loss, checkpoint, threshold hoặc siêu tham số. Cách dùng này tránh tự chia các khung hình liên quan của ClinicDB vào cả train và test, nhưng không tự chứng minh tính độc lập giữa hai nguồn.

Nếu không có metadata bệnh nhân/chuỗi nội soi, ghi rõ chỉ kiểm soát được các liên hệ phát hiện từ ảnh; không khẳng định split độc lập theo bệnh nhân.

**Sản phẩm:** bảng thống kê, hình mẫu, danh sách split, báo cáo kiểm tra trùng và nguồn dữ liệu.

### Bước 3 — Chốt preprocessing, huấn luyện và cách đánh giá chung

#### 3.1. Ảnh và mask

- Ảnh chuyển RGB. Mask chuyển grayscale rồi nhị phân hóa `pixel > 127` thành `0/1`; không dùng điều kiện `pixel > 0` cho mask JPG.
- CNN và SegFormer dùng khung đầu vào `256 × 256`: resize giữ tỷ lệ (làm tròn tới pixel) bằng bilinear cho ảnh, nearest-neighbor cho mask, rồi padding đặt ảnh ở giữa. Chia đều trái/phải hoặc trên/dưới; nếu tổng padding lẻ, phần dư 1 pixel nằm bên phải/dưới. Padding RGB màu đen trước normalization, padding mask bằng 0.
- Cấu hình khởi đầu dùng ảnh trong khoảng `[0,1]`, chuẩn hóa ImageNet: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; kiểm tra tính tương thích với checkpoint encoder đã chọn và giữ nhất quán trong các cặp ablation.
- Augmentation khởi đầu: lật ngang `p=0.5`, lật dọc `p=0.5`, xoay ngẫu nhiên một trong bốn góc `0/90/180/270°`. Áp dụng hình học đồng bộ ảnh–mask, chỉ ở train. Nếu thêm biến đổi màu, chỉ áp dụng lên ảnh và ghi tham số.
- Validation/test không augmentation ngẫu nhiên. Không tính thống kê chuẩn hóa hoặc chọn augmentation từ test.
- Dataset lưu geometry gồm kích thước gốc, kích thước nội dung sau resize, kích thước khung và padding `(left, top, right, bottom)`; trả valid_mask đánh dấu vùng ảnh thật. Augmentation hình học thực hiện trước resize/padding. Validation/test không augmentation nên có thể bỏ padding để đưa dự đoán về ảnh gốc. SAM dùng preprocessing riêng, không ép dùng letterbox của U-Net.

Độ phân giải lớn hơn hoặc augmentation khác chỉ được chọn trong giai đoạn phát triển bằng validation. Khóa thiết lập trước bảng so sánh chính; nếu muốn phân tích riêng độ phân giải, đặt mã thí nghiệm riêng.

#### 3.2. Cấu hình huấn luyện khởi đầu

Đây là **đề xuất của nhóm**, không phải cấu hình bắt buộc của cô hoặc cấu hình gốc của mọi mô hình.

| Thành phần | Giá trị khởi đầu |
| --- | --- |
| Input CNN/SegFormer | `256 × 256` |
| Batch size | `8`; kiểm tra VRAM trước khi chạy đầy đủ |
| Optimizer | Adam, learning rate `1e-4`, weight decay `0` |
| Epoch tối đa | `200` |
| Scheduler | Theo validation Dice, mode max, factor `0.5`, patience `5`, min LR `1e-6` |
| Early stopping | Validation Dice không cải thiện trong `20` epoch |
| Checkpoint | Validation Dice cao nhất; nếu bằng nhau chọn epoch sớm hơn |
| Fine-tuning | Train toàn bộ encoder và decoder ngay từ epoch đầu |
| Loss chung M1/M2/M3/A1 | `BCEWithLogitsLoss + soft Dice loss`, trọng số `1:1` |
| Seed huấn luyện | Chạy đầu với `42`; mục tiêu lặp `42, 43, 44` sau khi khóa cấu hình |

Giữ nguyên seed chia dữ liệu khi đổi seed huấn luyện. Ghi cách đặt seed, cấu hình deterministic nếu dùng và giới hạn tái lập theo môi trường.

Lưu config thực tế, nguồn/checksum pretrained weights, commit code, phiên bản thư viện, chế độ precision, batch size, thời gian và phần cứng theo từng run. Nếu giảm batch size hoặc đổi thiết lập vì VRAM, ghi rõ trước khi so sánh.

Các kiến trúc có thể cần LR khác nhau. Nếu tinh chỉnh riêng, dùng cùng ngân sách tìm kiếm hợp lý trên validation, công bố các cấu hình đã thử và khóa cấu hình được chọn trước khi test. Cặp ablation A1–M2 và nhóm loss phải giữ các yếu tố còn lại nhất quán.

#### 3.3. Dự đoán và metric

- Mô hình train xuất logits một kênh; target nhị phân có shape tương thích. Không sigmoid trước `BCEWithLogitsLoss`. Khi viết loss ở bước 4, dùng valid_mask để loại padding: BCE lấy tổng loss trên vùng hợp lệ chia số pixel hợp lệ, Dice/Tversky chỉ tổng hợp vùng hợp lệ; Focal cũng bỏ các pixel padding. Chuẩn bị mask không tự thay thế cho việc áp dụng nó trong loss.
- Khi đánh giá CNN/SegFormer, đưa logits decoder về khung đầu vào nếu cần, bỏ padding theo geometry của từng ảnh, rồi resize phần nội dung về kích thước gốc bằng bilinear với `align_corners=False`. Sau đó mới sigmoid và nhị phân hóa bằng `probability >= 0.5`. Không resize cả khung có padding về ảnh gốc.
- Giữ mask thật ở kích thước gốc. Tính Dice và IoU của lớp polyp trên từng ảnh, rồi lấy trung bình; không trộn với IoU nền hoặc thay bằng cách gộp toàn bộ pixel.
- Dice và IoU đánh giá tính từ mask nhị phân, phân biệt với soft Dice dùng để train.
- Nếu cả dự đoán và nhãn rỗng: Dice/IoU = 1; nếu chỉ một trong hai rỗng: Dice/IoU = 0. Precision/Recall bổ sung dùng giá trị 0 khi mẫu số bằng 0; công bố quy ước và số ca liên quan.
- Mặc định giữ threshold 0.5. Nếu muốn chọn threshold bằng validation, ghi thành cấu hình riêng và khóa trước khi test.
- Chọn best checkpoint bằng cùng cách tính validation Dice cho mọi loss. Chạy validation với chế độ eval và không tính gradient.

Kiểm tra metric bằng các trường hợp dễ biết: mask trùng hoàn toàn, không giao nhau, dự đoán rỗng, cả hai rỗng. Lưu kết quả từng ảnh để có thể tổng hợp lại bảng.

**Sản phẩm:** dataset/transforms, cấu hình run, metric và pipeline suy luận dùng chung.

### Bước 4 — Tự cài U-Net và hoàn thành M1

Tự viết các khối convolution, encoder/downsampling, bottleneck, decoder/upsampling, skip connection và lớp đầu ra bằng PyTorch; không dùng một mô hình U-Net dựng sẵn thay cho phần tự cài.

Kiến trúc khởi đầu: 4 lần downsampling; số kênh encoder `32/64/128/256`, bottleneck `512`; mỗi khối gồm hai convolution `3 × 3`, padding 1 và ReLU. Dùng max-pooling, transposed convolution để upsample, nối skip theo chiều kênh và convolution cuối `1 × 1` xuất một kênh logits. Nếu bổ sung BatchNorm hoặc thay khối, ghi cấu hình thực tế và hình kiến trúc.

Trước khi train đủ tập, cho mô hình học thử trên 4–8 ảnh, tắt augmentation và xem loss cùng overlay dự đoán để kiểm tra khả năng khớp dữ liệu nhỏ. Các lượt kiểm tra không được tính thành kết quả chính.

Sau khi pipeline hoạt động, chạy M1 và lưu checkpoint theo validation Dice. Milestone tuần 3 phải có kết quả baseline thực tế, không chỉ mô tả kiến trúc.

**Sản phẩm:** mã U-Net tự cài, hình kiến trúc, log và checkpoint M1.

### Bước 5 — Pretrained U-Net M2 và ablation A1

Dùng U-Net với encoder ResNet34 pretrained ImageNet, chẳng hạn qua thư viện `segmentation_models_pytorch`. Ghi thư viện/phiên bản, cấu hình decoder, nguồn weights và phần thư viện đảm nhiệm. Đầu ra một kênh logits, không có activation cuối trong mô hình train.

M2 so với M1 hoàn thành yêu cầu so sánh tối thiểu, nhưng khác cả kiến trúc lẫn pretraining. Vì vậy, nếu đủ ngân sách, chạy A1 với **đúng toàn bộ kiến trúc encoder–decoder của M2**, encoder khởi tạo ngẫu nhiên.

Trong cặp A1–M2, giữ split, preprocessing, augmentation, loss, lịch train và cách chọn checkpoint. Với từng seed, dùng cùng khởi tạo decoder để giảm yếu tố gây nhiễu; lưu cách khởi tạo. Chỉ thay nguồn khởi tạo encoder. Giữ chế độ fine-tuning toàn bộ ở cả hai.

**Sản phẩm:** M2, bảng so sánh M1–M2 và bảng A1–M2 nếu thực hiện.

### Bước 6 — SegFormer-B0 M3

Chọn SegFormer-B0 với encoder MiT-B0 pretrained ImageNet và đầu phân đoạn nhị phân khởi tạo mới. Ghi rõ mã nguồn triển khai, checkpoint encoder, cấu hình decoder và cách nạp weights.

Giữ nhãn `0 = nền`, `1 = polyp`; kiểm tra preprocessing không biến nền thành nhãn bỏ qua. Với thiết kế logits một kênh, dùng loss nhị phân chung của dự án và resize logits khi cần để khớp target; không mặc định dùng loss đa lớp của thư viện.

Dùng cùng split, độ phân giải ảnh đầu vào CNN/SegFormer, augmentation và quy tắc đánh giá. Báo cáo số tham số, thời gian và thiết lập fine-tuning.

Kết luận về các cấu hình đã thử trên dữ liệu này; không suy rộng một kết quả thành ưu thế của Transformer so với mọi CNN. Không cần thêm TransUNet để hoàn thành mục mở rộng kiến trúc.

**Sản phẩm:** M3 và bảng so sánh M1/M2/M3.

### Bước 7 — So sánh loss L1–L3

Cố định U-Net–ResNet34 pretrained và lần lượt chạy Dice, Binary Focal, Tversky. M2 dùng BCE + Dice là cấu hình tham chiếu bổ sung, không thay cho ba loss riêng.

Cấu hình khởi đầu:

| Loss | Định nghĩa và tham số cần ghi |
| --- | --- |
| BCE + Dice | BCEWithLogits lấy trung bình trên pixel + soft Dice loss, trọng số 1:1 |
| Soft Dice | `1 - (2 * sum(p*y) + eps) / (sum(p) + sum(y) + eps)`, `eps=1e-6` |
| Binary Focal | `-alpha_t * (1-p_t)^gamma * log(p_t)`; `alpha=0.25` cho foreground, `1-alpha` cho background; `gamma=2` |
| Tversky | `1 - (TP + eps) / (TP + alpha*FP + beta*FN + eps)`; `alpha=0.3`, `beta=0.7`, `eps=1e-6` |

Trong loss, `p = sigmoid(logits)`, `y` là mask 0/1; `p_t = p` nếu `y=1`, ngược lại `p_t = 1-p`. Với `v = valid_mask`, TP/FP/FN là các tổng mềm trên vùng ảnh thật: `sum(v*p*y)`, `sum(v*p*(1-y))`, `sum(v*(1-p)*y)`. Dice/Tversky tính trên từng ảnh rồi trung bình batch; BCE/Focal chỉ lấy trung bình pixel hợp lệ. Khi triển khai Focal, dùng dạng số học ổn định từ logits.

Các hệ số trên là đề xuất khởi đầu. Nếu tinh chỉnh, chỉ dùng validation và công bố phạm vi thử. Mỗi run phải khởi tạo lại từ cùng pretrained encoder và decoder ban đầu theo seed; không tiếp tục train từ checkpoint đã học với loss khác.

Giữ các yếu tố ngoài loss nhất quán, chọn checkpoint bằng validation Dice cho cả ba. Không so trực tiếp giá trị loss giữa các hàm để kết luận chất lượng.

**Sản phẩm:** bảng Dice/IoU/Precision/Recall, cấu hình loss và phân tích dự đoán thừa/bỏ sót.

### Bước 8 — SAM zero-shot S1–S2

Dùng SAM gốc ViT-B, đóng băng trọng số và không fine-tune trên Kvasir hoặc ClinicDB. Lưu nguồn, phiên bản và checksum checkpoint. Dùng preprocessing và phép chuyển tọa độ prompt theo triển khai SAM đã chọn.

Quy tắc prompt được cố định trước khi đánh giá:

| Mã | Prompt từ mask thật ở kích thước gốc | Quy tắc |
| --- | --- | --- |
| S1 | Một điểm foreground | Chọn pixel foreground có khoảng cách đến nền lớn nhất theo distance transform; coi ngoài ảnh là nền. Nếu hòa, chọn theo thứ tự hàng rồi cột; lưu tọa độ `(x,y)` |
| S2 | Bounding box | Box bao sát toàn bộ foreground, không nới hoặc thêm nhiễu; lưu `xmin, ymin, xmax, ymax` và quy ước tọa độ |

Đây là **oracle prompt**: sử dụng nhãn để cung cấp vị trí đối tượng. Mô tả là SAM không fine-tune có prompt từ nhãn, không xem là phân đoạn tự động chỉ từ ảnh.

Nếu SAM trả nhiều mask, chọn theo điểm chất lượng do SAM dự đoán; nếu bằng nhau chọn ứng viên đầu tiên. Không chọn theo IoU với mask thật và không sửa prompt dựa vào kết quả test.

Đưa mask về kích thước gốc theo hậu xử lý của SAM, tính cùng Dice/IoU như các mô hình khác. Nếu gặp mask thật rỗng, đánh dấu prompt oracle không áp dụng, công bố số ca và mẫu số đánh giá riêng; không âm thầm bỏ mẫu.

Trình bày S1/S2 ở bảng riêng, ghi rõ dữ liệu và thông tin bổ sung mà prompt cung cấp. Không gộp thứ hạng với U-Net/SegFormer mà bỏ qua khác biệt này.

**Sản phẩm:** bảng SAM, danh sách prompt, quy tắc chọn mask và hình minh họa.

### Bước 9 — Khóa cấu hình, đánh giá và phân tích

Trước khi dùng test, khóa split, preprocessing, cấu hình huấn luyện, loss, threshold, quy tắc SAM và danh sách thí nghiệm. Checkpoint của mỗi run chỉ được chọn bằng Kvasir validation.

Sau đó đánh giá trên Kvasir test và ClinicDB. Không dùng hai tập này để chọn lại mô hình hoặc siêu tham số. Nếu thay đổi phương pháp sau khi xem test, phải công bố việc đó và giới hạn của kết quả.

Với nhiều seed, giữ split cố định, lặp lại cấu hình đã khóa và báo cáo mean ± std giữa các run; không chỉ lấy seed tốt nhất. Lưu số ảnh, metric từng ảnh, mã run và checkpoint để tổng hợp lại.

Các bảng/hình dự kiến:

1. **Bảng kiến trúc:** M1/M2/M3 trên Kvasir test và ClinicDB; Dice/IoU, số tham số, thời gian và phần cứng.
2. **Bảng pretraining:** A1–M2 nếu thực hiện.
3. **Bảng loss:** L1/L2/L3, thêm M2 tham chiếu; Dice/IoU và Precision/Recall.
4. **Bảng SAM riêng:** S1/S2, checkpoint, loại prompt và số mẫu.
5. **Đồ thị:** train/validation loss, validation Dice; không so giá trị tuyệt đối giữa các loại loss.
6. **Hình đầu ra:** cùng ảnh cho các mô hình, có mask thật, mask dự đoán và overlay; phân tích ví dụ tốt, xấu và bất đồng.

Phân tích bổ sung theo kích thước polyp dùng `diện tích mask / diện tích ảnh gốc`. Chốt ngưỡng nhỏ/vừa/lớn từ phân vị 1/3 và 2/3 trên train Kvasir, áp dụng nguyên ngưỡng cho các tập đánh giá và báo số mẫu từng nhóm.

Giải thích các lỗi: polyp nhỏ, biên mờ, phản sáng, nếp gấp niêm mạc và thay đổi nguồn ảnh. Nếu chọn hình theo kết quả test, ghi quy tắc chọn, chẳng hạn ví dụ điển hình hoặc nhóm Dice thấp; không chỉ trình bày ảnh thuận lợi.

Attention map/Grad-CAM có thể bổ sung nếu phù hợp với câu hỏi; ưu tiên hình phân đoạn và phân tích lỗi rõ ràng. Ghi giới hạn về split theo ảnh, dữ liệu một lớp, số seed, ngân sách tinh chỉnh và tính phụ thuộc prompt của SAM.

**Sản phẩm:** CSV từng ảnh và bảng tổng hợp trong `results/tables/`, hình trong `results/figures/`, script/notebook tái tạo bảng chính.

### Bước 10 — Báo cáo, mã nguồn, slide và vấn đáp

#### Báo cáo theo yêu cầu của cô

- Tất cả các mốc dùng template ACL chính thức, tiếng Việt, khổ A4, hai cột; không chỉnh font/lề để nhồi nội dung.
- Dùng chế độ không ẩn danh `\usepackage[final]{acl}`, ghi họ tên và mã sinh viên của ba thành viên. Nếu dùng LaTeX, cấu hình tiếng Việt với XeLaTeX hoặc LuaLaTeX.
- Proposal cuối tuần 1 tối đa **2 trang**: bài toán, đầu vào/đầu ra, dữ liệu, baseline, metric và phân công.
- Milestone cuối tuần 3 tối đa **4 trang**: kết quả baseline thực tế, tiến độ, khó khăn và kế hoạch còn lại.
- Báo cáo cuối tuần 6 tối đa **8 trang nội dung**. References, Limitations, đóng góp thành viên và phụ lục không tính vào giới hạn theo đề; nội dung cốt lõi phải nằm trong phần chính.

Cấu trúc báo cáo cuối:

1. **Abstract, 150–250 từ:** bài toán, phương pháp và kết quả chính kèm số thực tế.
2. **Introduction:** động lực, đầu vào/đầu ra, câu hỏi và đóng góp.
3. **Related Work:** chỉ trích dẫn các tài liệu nhóm đã đọc.
4. **Method:** kiến trúc có hình, loss, nguồn weights, phần tự cài và phần dùng thư viện/mã có sẵn.
5. **Experiments:** dữ liệu, split, siêu tham số, phần cứng, thời gian, metric và bảng so với baseline.
6. **Analysis:** ablation, phân tích lỗi, trực quan hóa và giải thích kết quả.
7. **Conclusion:** kết quả chính và hướng phát triển.
8. **Limitations:** bắt buộc, nêu giới hạn dữ liệu, phương pháp và cách đánh giá.
9. **Đóng góp thành viên:** bắt buộc, nêu phần việc và tỷ lệ đóng góp ước tính.
10. **References:** trích dẫn tác giả–năm theo ACL, quản lý BibTeX nếu dùng LaTeX.
11. **Appendix:** tùy chọn; không chuyển kết quả cốt lõi sang phụ lục vì có thể không được đọc khi chấm.

Khai báo việc dùng AI; trích dẫn dữ liệu, bài báo, mã nguồn mở và pretrained weights. Tham khảo GitHub chính chủ không thay cho phần U-Net nhóm phải tự triển khai.

#### Mã nguồn và sản phẩm nộp

- Repo công khai hoặc mời cô vào; README hướng dẫn cài đặt, tải dữ liệu, train, đánh giá và tái tạo bảng chính.
- Cập nhật `requirements.txt` hoặc `environment.yml` theo môi trường train thực tế; lưu config và ID split lên Git.
- Checkpoint/dữ liệu lớn dùng link ngoài. Log W&B/TensorBoard phải chia sẻ được với cô; việc `logs/` bị Git ignore không thay cho yêu cầu chia sẻ log.
- Slide **10–12 phút**. Mục sản phẩm ghi demo nếu có, còn rubric có tiêu chí demo chạy được; nhóm dự kiến chuẩn bị demo gọn ảnh → mask/overlay, không cần xây ứng dụng lớn.
- Demo suy luận không phụ thuộc mask thật; nếu demo SAM oracle thì giải thích rõ nguồn prompt.
- Nộp PDF tên `NhomXX_DeTai02.pdf`; kiểm tra quyền truy cập repo, weights và log trước khi nộp.

#### Quá trình và vấn đáp

Duy trì commit và log trong quá trình thực hiện. Đề phân bổ: kỹ thuật/thí nghiệm 25%, phân tích/báo cáo 25%, quá trình 10%, thuyết trình/demo 10%, vấn đáp cá nhân 30%.

Mỗi người phải giải thích được code mình làm, phương pháp, kết quả và ít nhất phần liên quan của bạn khác; đọc chéo cả code dùng thư viện hoặc do AI hỗ trợ. Chuẩn bị trả lời về shape, mask, loss, skip connection, pretraining, lựa chọn checkpoint và prompt SAM; luyện sửa một đoạn nhỏ tại chỗ.

**Ngưỡng vấn đáp:** dưới 4/10 thì tổng điểm bài tập lớn của cá nhân không vượt 5/10. Đề không chấm theo độ chính xác cao hay thấp; kết quả không vượt baseline vẫn có thể được điểm tốt khi thí nghiệm đúng và phân tích được nguyên nhân.

## 4. Lịch triển khai 6 tuần

| Tuần | Công việc dự kiến | Sản phẩm/mốc |
| --- | --- | --- |
| 1 | Đọc bài nền tảng; kiểm tra data, split, môi trường; phân công; mời cô vào repo và thiết lập log chia sẻ | **Proposal ACL ≤ 2 trang, cuối tuần 1** |
| 2 | Hoàn thiện dataset, metric, U-Net tự cài; học thử vài ảnh; train M1 | Baseline, checkpoint và log |
| 3 | Train M2, triển khai/chạy thử M3; tổng hợp baseline, tiến độ và khó khăn | **Milestone ACL ≤ 4 trang, cuối tuần 3** |
| 4 | Chạy M3 đầy đủ, L1–L3 và A1 khi đủ ngân sách; chọn cấu hình bằng validation | Bảng run, cấu hình được chọn, chuẩn bị mã đánh giá SAM |
| 5 | Lặp seed theo ngân sách; khóa thiết lập; đánh giá test/ClinicDB và SAM; phân tích lỗi, trực quan hóa | Bảng/hình và bản nháp báo cáo; bản nháp không chấm điểm |
| 6 | Hoàn thiện báo cáo, mã tái tạo bảng, slide, demo; đọc chéo và luyện vấn đáp | **Báo cáo cuối, mã nguồn, slide và thuyết trình** |

Ưu tiên hoàn tất M1/M2 và đánh giá đúng trước khi mở rộng. Nếu thiếu thời gian, điều chỉnh các phần nhóm tự bổ sung như A1 hoặc số lượt lặp, công bố phần chưa làm; không bỏ yêu cầu tối thiểu hoặc các mốc nộp bài.

## 5. Nguồn tham khảo và cách dùng

Ưu tiên bài báo gốc và repo chính thức. Khi thực sự sử dụng một nguồn, lưu phiên bản/commit, checkpoint và trích dẫn trong báo cáo; các link dưới đây không có nghĩa repo hiện đã tích hợp các mô hình.

- [U-Net, bài báo gốc](https://arxiv.org/abs/1505.04597): cơ sở tự cài; ghi rõ các thay đổi so với kiến trúc gốc.
- [Kvasir-SEG, GitHub tác giả](https://github.com/DebeshJha/Kvasir-SEG) và [trang dữ liệu Simula](https://datasets.simula.no/kvasir-seg/): nguồn dữ liệu; nếu chọn split khác tác giả, công bố split của nhóm.
- [SegFormer, repo chính thức NVlabs](https://github.com/NVlabs/SegFormer): tham khảo kiến trúc và nguồn pretrained encoder.
- [SAM, repo chính thức Meta](https://github.com/facebookresearch/segment-anything): checkpoint, preprocessing, prompt và hậu xử lý.
- [Segmentation Models PyTorch, repo thư viện](https://github.com/qubvel-org/segmentation_models.pytorch): nguồn thư viện dự kiến cho U-Net–ResNet34.

Không lấy checkpoint đã fine-tune trên Kvasir/ClinicDB làm khởi tạo rồi mô tả là chỉ pretrained ImageNet hoặc đánh giá ngoài độc lập. Kết quả của paper dùng split, dữ liệu train, preprocessing hoặc prompt khác chỉ là tham khảo; không coi là so sánh trực tiếp với bảng của nhóm.
