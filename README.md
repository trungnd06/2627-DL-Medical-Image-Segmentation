# Medical Image Segmentation

Đề tài 2 môn Học sâu: phân đoạn polyp trong ảnh nội soi. Đầu vào là ảnh RGB; đầu ra là mask cùng kích thước ảnh gốc, với `1 = polyp`, `0 = nền`. Chỉ số đánh giá chính: **Dice và IoU**.

Nhóm dự kiến so sánh U-Net tự cài, U-Net có pretrained encoder và SegFormer, đồng thời khảo sát loss và SAM zero-shot. Kế hoạch chi tiết nằm trong [pipeline.md](pipeline.md).

**Tổ chức code:** module Python chứa logic; notebook gọi module, giải thích bằng Markdown và hiển thị hình. Script là cách chạy thay thế từ terminal.

**Trạng thái:** đã có chuẩn bị dữ liệu, split cố định, Dataset, transforms, DataLoader và metric. Mô hình, loss và vòng train/evaluate chưa triển khai. Các mục `[khung]` bên dưới chưa có chức năng chạy.

## 1. Cấu trúc dự án

Cây dưới mô tả mã nguồn và các đầu vào cần chia sẻ trên Git. Riêng `data/` là dữ liệu cục bộ được tạo sau khi tải, không đưa lên Git. Không liệt kê các báo cáo/hình sinh tự động hoặc file làm việc riêng.

```text
Medical-Image-Segmentation/
├── README.md                         # Cấu trúc, cài đặt và hướng dẫn chạy
├── pipeline.md                       # Kế hoạch thực hiện và bảng thí nghiệm
├── requirements.txt                  # Thư viện chuẩn bị dữ liệu, notebook 01–02
├── requirements-training.txt         # Bao gồm requirements.txt, thêm PyTorch và YAML
├── .gitignore                        # Quy tắc bỏ qua file cục bộ/sinh tự động
├── notebooks/                        # Giải thích, gọi module và xem hình minh họa
│   ├── 01_download_data.ipynb         # Tải/xác minh nguồn và xem mẫu ảnh–mask
│   ├── 02_prepare_data.ipynb          # Thống kê, rà ảnh gần trùng và chia tập
│   └── 03_data_pipeline.ipynb         # Kiểm tra batch, transforms và Dice/IoU
├── src/                              # Module dùng chung
│   ├── __init__.py                   # Khai báo package src
│   ├── datasets/
│   │   ├── __init__.py               # Khai báo package dữ liệu
│   │   ├── common.py                 # Đường dẫn, hằng số, CSV và checksum
│   │   ├── download.py               # Tải/đối chiếu dữ liệu từ nguồn cố định
│   │   ├── statistics.py             # Kiểm tra cặp, thống kê và overlay nhãn
│   │   ├── duplicates.py             # Hash ảnh, sàng lọc và đọc quyết định rà
│   │   ├── splitting.py              # Chia theo nhóm/seed, kiểm tra và lưu split
│   │   ├── reporting.py              # Xuất bảng/hình và tổng kết chuẩn bị dữ liệu
│   │   ├── preparation.py            # Điều phối bước chuẩn bị và chia tập
│   │   ├── dataset.py                # Dataset, ghép batch và tạo DataLoader
│   │   ├── transforms.py             # Resize, augmentation đồng bộ, normalization
│   │   └── preview.py                # Minh họa ảnh–mask trước/sau transforms
│   ├── models/
│   │   ├── unet.py                   # [khung] U-Net tự cài, M1
│   │   ├── pretrained_unet.py        # [khung] U-Net–ResNet34, M2/A1
│   │   ├── segformer.py              # [khung] SegFormer-B0, M3
│   │   └── sam.py                    # [khung] SAM zero-shot, S1/S2
│   ├── metrics.py                    # Dice/IoU/Precision/Recall theo từng ảnh
│   ├── utils.py                      # Đặt seed và đọc config YAML
│   ├── losses.py                     # [khung] BCE+Dice, Focal, Tversky
│   └── engine.py                     # [khung] Vòng train/validation
├── scripts/                          # Các lệnh chạy từ terminal
│   ├── __init__.py                   # Hỗ trợ python -m scripts.<tên>
│   ├── prepare_data.py               # Chuẩn bị/chia tập; --download để tải trước
│   ├── preview_data.py               # Kiểm tra đầu vào và xuất hình transforms
│   ├── train.py                      # [khung] Khởi chạy huấn luyện
│   ├── evaluate.py                   # [khung] Đánh giá mô hình đã train
│   ├── evaluate_sam.py               # [khung] Đánh giá SAM có prompt
│   └── predict.py                    # [khung] Dự đoán mask từ ảnh
├── configs/                          # Thông số thí nghiệm
│   ├── unet.yaml                     # Preprocessing/loader và cấu hình M1 dự kiến
│   ├── pretrained_unet.yaml          # [khung] Cấu hình M2/A1
│   ├── segformer.yaml                # [khung] Cấu hình M3
│   └── sam.yaml                      # [khung] Cấu hình SAM/prompt
├── splits/                           # Đầu vào cố định, dùng chung cho cả nhóm
│   ├── kvasir_train.txt              # 700 ID train
│   ├── kvasir_val.txt                # 150 ID validation
│   ├── kvasir_test.txt               # 150 ID test nội bộ
│   ├── clinicdb_external.txt         # 612 ID đánh giá ngoài
│   └── duplicate_review.csv          # Quyết định rà ảnh để tái tạo cách chia
├── tests/                            # Kiểm thử code, khác tập dữ liệu test
│   ├── test_data_preparation.py      # Kiểm tra nguồn, chia tập và chống ghi đè
│   └── test_training_data.py         # Kiểm tra transforms, loader và metric
├── report/                           # Báo cáo ACL tiếng Việt
│   ├── main.tex                      # [khung] Nội dung báo cáo
│   └── references.bib                # [khung] Trích dẫn BibTeX
└── data/                             # Tạo tại máy sau khi tải; không đưa lên Git
    ├── Kvasir-SEG/
    │   ├── images/                   # 1.000 ảnh JPG
    │   └── masks/                    # 1.000 mask JPG, cùng tên với ảnh
    ├── CVC-ClinicDB/
    │   ├── images/                   # 612 ảnh PNG
    │   └── masks/                    # 612 mask PNG, cùng tên với ảnh
    ├── _downloads/                   # ZIP/snapshot cache để tái sử dụng
    └── dataset_manifest.json         # Nguồn, phiên bản và checksum ảnh/mask
```

## 2. Cài môi trường

Khuyến nghị Python **3.10 hoặc 3.12**. Tạo và kích hoạt môi trường riêng cho dự án:

```powershell
conda create -n medical-seg python=3.12 -y
conda activate medical-seg
```

### Vì sao có hai file requirements?

| File | Dùng khi nào? |
| --- | --- |
| `requirements.txt` | Chỉ tải/chuẩn bị dữ liệu, chạy notebook 01–02; không cần PyTorch |
| `requirements-training.txt` | Chạy đầy đủ phần hiện có, gồm notebook 03; tự bao gồm file trên rồi thêm torch và PyYAML |

**Không cần cài lần lượt cả hai.** Để chạy toàn bộ dự án hiện tại, chọn bản PyTorch phù hợp rồi cài `requirements-training.txt`. Tên file training chỉ nhóm thư viện phục vụ giai đoạn này, không có nghĩa vòng train đã hoàn thành.

Máy có NVIDIA và driver tương thích CUDA 12.6:

```powershell
python -m pip install torch==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-training.txt
```

Máy không có GPU NVIDIA có thể dùng CPU để chạy notebook và kiểm tra dữ liệu:

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
```

Chọn **một** bản CPU hoặc CUDA phù hợp với phần cứng và driver. Xem [các bản cài PyTorch chính thức](https://pytorch.org/get-started/previous-versions/#v291). Các transforms hiện tại chưa cần torchvision.

Nếu chỉ chạy notebook 01–02: `python -m pip install -r requirements.txt` là đủ.

## 3. Chạy bằng notebook

Trong VS Code, mở notebook bằng **Jupyter Notebook**, chọn **Select Kernel** đúng môi trường đã cài, rồi **Restart Kernel → Run All**. Hoặc chạy `python -m jupyterlab` tại gốc repo và mở notebook trong JupyterLab.

| Thứ tự | Notebook | Công việc |
| --- | --- | --- |
| 01 | [01_download_data.ipynb](notebooks/01_download_data.ipynb) | Tải/xác minh nguồn, ghép ảnh–mask và xem mẫu |
| 02 | [02_prepare_data.ipynb](notebooks/02_prepare_data.ipynb) | Thống kê, sàng lọc gần trùng, giữ nhóm và tạo split |
| 03 | [03_data_pipeline.ipynb](notebooks/03_data_pipeline.ipynb) | Đọc batch, xem augmentation và kiểm tra metric |

**Máy mới:** lấy đủ mã nguồn và đầu vào trong splits/ từ Git, cài thư viện rồi chạy **01 → 02 → 03**. Notebook 01 cần Internet để truy vấn nguồn; 02–03 dùng dữ liệu cục bộ. Không cần GPU cho bước tải/chuẩn bị.

**Đã có dữ liệu và split:** có thể chạy thẳng notebook 03. Không cần tải hoặc chia lại dữ liệu trước mỗi lần làm bài.

Số notebook là thứ tự chạy. Bước 1 trong pipeline.md là chốt phạm vi/phân công; notebook 01–02 phục vụ chuẩn bị dữ liệu của bước 2, notebook 03 phục vụ bước 3.

Kết quả mong đợi: **700 train / 150 validation / 150 test**, **612 external**. Batch ảnh **[8,3,256,256]**, mask **[8,1,256,256]** chỉ chứa 0/1. Hình minh họa dùng mask thật; ví dụ metric chưa phải kết quả của mô hình.

## 4. Chạy thay thế từ terminal

Đứng tại gốc repo và kích hoạt môi trường. Không cần chạy thêm notebook cho cùng công việc.

```powershell
# Chưa có dữ liệu: tải/xác minh trước, rồi tự chuẩn bị và chia tập
python -m scripts.prepare_data --project-root . --download

# Đã có dữ liệu: chỉ kiểm tra/tái tạo bước chuẩn bị
python -m scripts.prepare_data --project-root .

# Kiểm tra Dataset, transforms, batch và metric
python -m scripts.preview_data --project-root .
```

Hai lệnh prepare_data là lựa chọn theo tình trạng dữ liệu, không cần chạy cả hai. Các script tự gọi module đúng thứ tự; không chạy từng file trong src/.

Bảng, hình và báo cáo được code tự tạo cục bộ khi chạy; không cần chép output terminal hoặc đưa chúng lên Git. Notebook hiển thị hình trực tiếp, terminal in vị trí đã lưu để mở xem.

Kiểm thử sau khi cài đầy đủ thư viện:

```powershell
python -m unittest discover -s tests -v
```

Chỉ kiểm thử phần chuẩn bị dữ liệu: `python -m unittest discover -s tests -p test_data_preparation.py -v`.

## 5. Dữ liệu và quy tắc dùng chung

| Dataset | Số cặp | Định dạng | Vai trò |
| --- | ---: | --- | --- |
| Kvasir-SEG | 1.000 | JPG, độ phân giải thay đổi | Train/validation/test nội bộ |
| CVC-ClinicDB | 612 | PNG, 384×288 | Đánh giá ngoài, không tinh chỉnh |

- **Kvasir:** [trang công bố Simula](https://datasets.simula.no/kvasir-seg/), [ZIP gốc](https://datasets.simula.no/downloads/kvasir-seg.zip). SHA-256 ZIP khóa trong download.py: `03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7`.
- **ClinicDB:** [nguồn công bố](https://polyp.grand-challenge.org/CVCClinicDB/); tải từ [berkaytrhn/cvc-clinicdb](https://huggingface.co/datasets/berkaytrhn/cvc-clinicdb) với revision `07f1aae53befabc313f9435c8b1b9565642a36c8`. Module ánh xạ Original/Ground Truth sang images/masks, giữ nguyên byte.
- Không ghi đè dữ liệu gốc khi resize. Mask chuyển grayscale rồi nhị phân hóa >127. Ảnh resize bilinear, mask nearest-neighbor; augmentation hình học đồng bộ và chỉ ở train. Tensor sau xử lý nằm trong RAM để mô hình sử dụng.
- Cả nhóm dùng chung ID trong splits/, seed chia 42. Những file này và quyết định rà ảnh phải có trên Git; không chỉ giữ trong máy một người. Mỗi dòng split là ID không có đuôi ảnh.
- Hash chỉ sàng lọc, unsure không có nghĩa là duplicate đã xác nhận. Giữ chung các quan hệ confirmed/unsure và công bố giới hạn thiếu metadata bệnh nhân. Quyết định rà đã lưu được đọc lại khi chạy trên máy khác.
- Nếu có cặp chưa rà trong Kvasir/cross-dataset, notebook 02 sẽ yêu cầu cập nhật quyết định trước khi chia. Split đã có nội dung khác sẽ làm chương trình dừng; không tự xóa danh sách để vượt kiểm tra.
- Chọn checkpoint, loss và threshold bằng validation Kvasir; không dùng test/ClinicDB để tinh chỉnh. Loader mặc định chỉ đọc train/validation; thêm test/external khi đánh giá cuối.
- Metric tính từng ảnh rồi trung bình theo số ảnh. Resize logits về kích thước gốc trước sigmoid/threshold 0.5, so với mask thật gốc; không trung bình các batch mà bỏ qua số ảnh khác nhau ở batch cuối.
- Trước khi commit notebook, dùng **Clear All Outputs** và lưu lại. Giữ code/Markdown trên Git, để mỗi thành viên tự tạo hình và kết quả bằng Run All.

## 6. Phần tiếp theo

Triển khai **U-Net tự cài → BCE+Dice → vòng train/validation → học thử 4–8 ảnh → train M1 đầy đủ**. Chưa có lệnh train hoạt động; phần training trong unet.yaml hiện là thông số khởi đầu.

Theo đề bài: tối thiểu U-Net tự cài, so sánh pretrained U-Net, Dice/IoU; mở rộng SegFormer hoặc TransUNet, so sánh Dice/Focal/Tversky và SAM zero-shot. Đối chiếu tiến độ, báo cáo ACL, phân công và các mốc nộp trong pipeline.md.

## 7. Tài liệu tham khảo dữ liệu

1. Jha, D., et al. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. MMM, 451–462. [Bài báo](https://arxiv.org/abs/1911.07069).
2. Bernal, J., et al. (2015). *WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians*. Computerized Medical Imaging and Graphics, 43, 99–111. [Nguồn và trích dẫn](https://polyp.grand-challenge.org/CVCClinicDB/).
