# Medical Image Segmentation

Bài tập lớn môn Học sâu: phân đoạn polyp trong ảnh nội soi. Đầu vào là ảnh RGB; đầu ra là mask cùng kích thước ảnh, với `1 = polyp`, `0 = nền`.

Hướng thực nghiệm dự kiến: tự cài U-Net, so sánh với U-Net có encoder pretrained và SegFormer; khảo sát Dice/Focal/Tversky loss, khả năng tổng quát hóa sang nguồn ảnh khác và SAM zero-shot. Chỉ số bắt buộc là **Dice và IoU**.

**Trạng thái hiện tại:** repo đang ở bước chuẩn bị dữ liệu. Notebook tải dữ liệu đã được triển khai; các file mô hình, train/evaluate, cấu hình huấn luyện và danh sách split hiện là khung chờ hoàn thiện. Chưa có kết quả huấn luyện để báo cáo.

## 1. Chạy nhanh sau khi clone repo

Yêu cầu: **Python 3.10 trở lên**, Internet, khoảng **1 GB dung lượng trống** để chứa dữ liệu, ZIP và snapshot cache. Bước tải dữ liệu không cần GPU, tài khoản Hugging Face hoặc API key; repo dữ liệu Hugging Face đang công khai.

Mở terminal tại thư mục repo vừa clone. Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Trên Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

Trong JupyterLab, mở **[src/datasets/downdata.ipynb](src/datasets/downdata.ipynb)**, chọn kernel của môi trường vừa cài và **Run → Run All Cells**. Nếu dùng VS Code, mở repo, chọn đúng Python interpreter/kernel rồi chạy **Run All** trong notebook.

Notebook sẽ:

1. Tìm thư mục gốc repo từ thư mục hiện tại hoặc các thư mục cha.
2. Tạo các thư mục cục bộ `data/`, `checkpoints/` và `logs/` nếu chưa có.
3. Tải ZIP **Kvasir-SEG từ Simula** và snapshot **CVC-ClinicDB từ `berkaytrhn/cvc-clinicdb` trên Hugging Face**. Khóa checksum ZIP và revision snapshot; cache nằm trong `data/_downloads/`.
4. Lấy đúng các cặp ảnh–mask vào cấu trúc chuẩn, giữ nguyên nội dung JPG/PNG, không resize hoặc sửa nhãn.
5. Kiểm tra số lượng, tên file, khả năng đọc ảnh, kích thước ảnh–mask và mask có vùng polyp sau khi áp dụng ngưỡng 127.
6. Lưu nguồn tải và checksum từng file vào `data/dataset_manifest.json`, hiển thị bảng số lượng và ảnh mẫu.

Chạy lại notebook sẽ kiểm tra và tái sử dụng dữ liệu khớp nguồn; nếu thiếu file thì bổ sung. CVC-ClinicDB được đối chiếu với SHA-256 từng PNG trong metadata LFS của revision đã khóa. Nếu phát hiện file đã tồn tại nhưng khác nguồn hoặc có file ảnh ngoài danh sách dự kiến, notebook dừng và chỉ rõ xung đột để bạn kiểm tra, **không tự ghi đè hoặc xóa dữ liệu đó**. Cần Internet để truy vấn metadata Hugging Face khi chạy lại; file ảnh đã đủ và đúng checksum sẽ không tải lại.

Có thể chạy toàn bộ notebook từ terminal mà không lưu output vào notebook trong repo:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute src/datasets/downdata.ipynb --ExecutePreprocessor.timeout=1800 --output downdata.executed.ipynb --output-dir logs
```

Lệnh trên dùng trong PowerShell tại thư mục gốc repo. Nếu chạy notebook từ bên ngoài cây thư mục repo, điền đường dẫn repo vào biến `PROJECT_ROOT_OVERRIDE` ở cell cấu hình.

Nếu gặp lỗi:

- `ModuleNotFoundError`: kiểm tra kernel có dùng đúng môi trường đã cài `requirements.txt` không.
- Lỗi kết nối: chạy lại; Kvasir loại bỏ file ZIP tải dở, Hugging Face quản lý cache của snapshot. File đã đủ và kiểm tra thành công được dùng lại.
- Lỗi checksum hoặc file xung đột: kiểm tra đường dẫn được báo; nếu file tải bị hỏng, di chuyển riêng file đó ra ngoài rồi chạy lại. Không xóa toàn bộ `data/` hoặc tắt kiểm tra checksum.
- Lỗi chứng chỉ HTTPS: cập nhật Python/chứng chỉ của môi trường hoặc cấu hình chứng chỉ tin cậy của mạng đang dùng. Notebook dùng xác minh HTTPS mặc định, không tắt TLS verification.

## 2. Hai bộ dữ liệu sử dụng

| Dataset | Số cặp ảnh–mask | Định dạng trong repo | Vai trò dự kiến |
| --- | ---: | --- | --- |
| **Kvasir-SEG** | 1.000 | `images/*.jpg`, `masks/*.jpg`; độ phân giải thay đổi | Train, validation và test nội bộ |
| **CVC-ClinicDB** | 612 | `images/1.png` … `612.png`, `masks/1.png` … `612.png`; `384 × 288` | Đánh giá bên ngoài, không dùng tinh chỉnh mô hình |

### Kvasir-SEG

- Thông tin và điều kiện sử dụng: <https://datasets.simula.no/kvasir-seg/>.
- ZIP chính thức: <https://datasets.simula.no/downloads/kvasir-seg.zip>.
- Notebook chỉ lấy hai thư mục ảnh và mask phục vụ phân đoạn; không cần bounding box JSON.

### CVC-ClinicDB

- Thông tin, bản quyền và trích dẫn gốc: <https://polyp.grand-challenge.org/CVCClinicDB/>.
- Nguồn tải tự động: **đúng repo Hugging Face mà nhóm đã sử dụng**, <https://huggingface.co/datasets/berkaytrhn/cvc-clinicdb>.
- Notebook dùng `huggingface_hub.snapshot_download` với `repo_id="berkaytrhn/cvc-clinicdb"`, `repo_type="dataset"` và **revision cố định `07f1aae53befabc313f9435c8b1b9565642a36c8`**. Đây là snapshot đã đối chiếu với dữ liệu hiện tại của nhóm.
- Lấy đầy đủ **612 cặp PNG** trong `CVC-ClinicDB/Original/` và `CVC-ClinicDB/Ground Truth/`; kiểm tra SHA-256 của cả 1.224 file theo metadata LFS ở revision đó. Không tự chuyển sang mirror khác khi tải lỗi.
- Đã đối chiếu cả **1.224 file ảnh/mask hiện tại của nhóm** với SHA-256 của revision này: tất cả đều khớp. Việc chuẩn hóa thư mục không thay đổi dữ liệu.
- Cache snapshot tại `data/_downloads/clinicdb-hf/` (đường dẫn ngắn để thuận tiện trên Windows); revision vẫn được khóa trong lệnh tải và ghi vào manifest. Khi tải dữ liệu mới, notebook cũng giữ `README.md`, `metadata.csv` và `class_dict.csv` từ nguồn trong cache; tên ảnh được giữ nguyên.

Lệnh bạn đã dùng để tải CVC-ClinicDB:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='berkaytrhn/cvc-clinicdb', repo_type='dataset', local_dir='data/CVC-ClinicDB')"
```

Lệnh này tải repo và giữ cấu trúc nguồn, nên ảnh nằm tại `data/CVC-ClinicDB/CVC-ClinicDB/Original/` và mask tại `data/CVC-ClinicDB/CVC-ClinicDB/Ground Truth/`; nó **không tự tạo `images/` và `masks/`**. Nó cũng mặc định lấy nhánh `main`, có thể thay đổi về sau. Các thành viên nên chạy notebook để dùng cùng revision và nhận sẵn cấu trúc chuẩn:

| Đường dẫn trong snapshot Hugging Face | Đường dẫn dùng trong dự án |
| --- | --- |
| `CVC-ClinicDB/Original/<tên>.png` | `data/CVC-ClinicDB/images/<tên>.png` |
| `CVC-ClinicDB/Ground Truth/<tên>.png` | `data/CVC-ClinicDB/masks/<tên>.png` |

Notebook sao chép **nguyên byte**, chỉ đổi cách sắp xếp thư mục. Nếu trước đó đã tải bằng lệnh trên, notebook vẫn chuẩn bị thư mục chuẩn mà không xóa cây snapshot cũ.

Checksum ZIP Kvasir-SEG và revision CVC-ClinicDB được ghi trực tiếp trong cell cấu hình của notebook. Không tùy tiện bỏ kiểm tra checksum hoặc thay revision; cần xác minh lại dữ liệu trước khi đổi nguồn/phiên bản.

Hai bộ đều dành cho nghiên cứu và giáo dục; điều kiện của tác giả gốc vẫn áp dụng khi tải từ mirror. Hãy trích dẫn các bài báo ở cuối README trong báo cáo.

## 3. Cấu trúc thư mục sau khi chạy notebook

```text
Medical-Image-Segmentation/
├── data/                              # Tạo tại máy, Git bỏ qua
│   ├── Kvasir-SEG/
│   │   ├── images/                    # 1.000 JPG
│   │   └── masks/                     # 1.000 JPG, cùng tên với ảnh
│   ├── CVC-ClinicDB/
│   │   ├── images/                    # 612 PNG
│   │   └── masks/                     # 612 PNG, cùng tên với ảnh
│   ├── _downloads/                    # ZIP Kvasir và cache snapshot Hugging Face
│   └── dataset_manifest.json          # Nguồn tải, thống kê, checksum file
├── src/
│   ├── datasets/
│   │   ├── downdata.ipynb             # Notebook cần chạy để tái tạo data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── unet.py
│   │   ├── pretrained_unet.py
│   │   ├── segformer.py
│   │   └── sam.py
│   ├── losses.py
│   ├── metrics.py
│   ├── engine.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_sam.py
│   └── predict.py
├── configs/                           # Cấu hình thí nghiệm, đưa lên Git
├── splits/                            # Danh sách chia tập, đưa lên Git
├── results/
│   ├── tables/
│   └── figures/
├── checkpoints/                       # Trọng số, Git bỏ qua
├── logs/                              # Log, Git bỏ qua
├── report/                            # Báo cáo ACL
├── requirements.txt
├── README.md
└── .gitignore
```

Git không lưu thư mục rỗng. Vì thế một số thư mục như `results/tables/` và `results/figures/` có thể chưa xuất hiện sau khi clone; tạo khi bắt đầu xuất kết quả. `checkpoints/` và `logs/` được notebook tạo lại; `.venv/` được tạo bằng lệnh cài đặt ở trên.

## 4. Vì sao không thấy `data/` trên GitHub?

`.gitignore` hiện bỏ qua các đường dẫn sau:

| Quy tắc | Nội dung không đưa lên Git | Cách tái tạo/sử dụng |
| --- | --- | --- |
| `data/` | Hai dataset, ZIP/snapshot cache và manifest cục bộ | Chạy `src/datasets/downdata.ipynb` |
| `checkpoints/` | Trọng số mô hình | Tạo khi train; chia sẻ trọng số lớn bằng link ngoài |
| `logs/` | TensorBoard/W&B và log thực nghiệm cục bộ | Tạo khi chạy; chia sẻ log với giảng viên theo yêu cầu |
| `__pycache__/` | Cache Python | Python tự tạo |
| `.venv/` | Môi trường ảo | Cài lại từ `requirements.txt` |
| `.ipynb_checkpoints/` | Bản lưu tự động của Jupyter | Jupyter tự tạo |

**Đưa lên Git:** notebook tải dữ liệu, code, cấu hình, `requirements.txt`, README, danh sách split đã chốt, bảng CSV và hình kết quả chọn lọc. `results/` không bị ignore.

Không dùng `git add -f data/` để đẩy dữ liệu. `.gitignore` chỉ bỏ qua file chưa được theo dõi; nếu trước đó đã commit dữ liệu thì chỉ thêm quy tắc ignore sẽ không làm dữ liệu biến mất khỏi Git.

## 5. Lưu ý trước khi triển khai huấn luyện

- Notebook **chưa tạo hoặc thay đổi các file trong `splits/`**. Nhóm phải kiểm tra trùng/gần trùng và chốt danh sách chia tập trước khi train. Tỉ lệ Kvasir dự kiến là 70/15/15; ghi rõ giới hạn nếu không có metadata bệnh nhân/chuỗi nội soi để chia theo nhóm.
- Giữ CVC-ClinicDB làm tập đánh giá bên ngoài. Chọn checkpoint, loss, ngưỡng và siêu tham số trên validation của Kvasir-SEG.
- Mask nguồn có thể là ảnh RGB và có mức xám ngoài `0/255` (đặc biệt JPG). Khi tạo tensor: chuyển grayscale rồi nhị phân hóa `pixel > 127` thành `0/1`. Không coi mọi pixel `> 0` là polyp. Notebook chỉ kiểm tra/hiển thị với ngưỡng này, giữ file nguồn nguyên vẹn.
- Phép biến đổi hình học phải đồng bộ ảnh–mask; mask resize bằng nearest-neighbor; augmentation chỉ áp dụng ở train.
- SAM zero-shot phải giữ trọng số cố định và khai báo nguồn prompt. Nếu điểm/box được tạo từ mask thật, ghi rõ **oracle prompt**, trình bày riêng với mô hình phân đoạn tự động.
- `requirements.txt` hiện phục vụ bước dữ liệu/notebook. Thư viện train như PyTorch và các thư viện mô hình sẽ được bổ sung theo môi trường GPU khi triển khai; hiện chưa có lệnh train/evaluate hoạt động.

## 6. Các yêu cầu nộp bài cần theo dõi

Theo file đề bài của môn học:

- Nhóm 3 sinh viên, thực hiện trong 6 tuần. Tạo repo, mời giảng viên và chia sẻ log huấn luyện từ quá trình thực hiện.
- Tối thiểu: tự cài U-Net; so sánh U-Net pretrained encoder; báo cáo Dice và IoU.
- Mở rộng: SegFormer **hoặc** TransUNet; so sánh Dice/Focal/Tversky loss; đánh giá SAM zero-shot. Kế hoạch của repo chọn SegFormer và thử cả ba hướng mở rộng.
- Cuối tuần 1: proposal tối đa 2 trang. Cuối tuần 3: milestone tối đa 4 trang. Tuần 6: bản cuối tối đa 8 trang nội dung, theo template ACL, viết tiếng Việt.
- Báo cáo có mô tả phần tự cài/dùng thư viện, thiết lập thực nghiệm, ablation, phân tích lỗi, Limitations, đóng góp thành viên và khai báo sử dụng AI. Giữ đầy đủ trích dẫn nguồn dữ liệu/mã/pretrained weights.
- Có hướng dẫn cài đặt, tải dữ liệu, train và đánh giá; script/notebook tái tạo bảng chính; slide 10–12 phút và demo nếu có. Hướng dẫn train/evaluate sẽ được cập nhật khi có triển khai thực tế.
- Vấn đáp cá nhân chiếm 30%; dưới 4/10 thì tổng điểm bài tập lớn của sinh viên đó không vượt quá 5/10.

## 7. Tài liệu tham khảo dữ liệu

1. Jha, D., et al. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. International Conference on Multimedia Modeling, 451–462. <https://arxiv.org/abs/1911.07069>.
2. Bernal, J., et al. (2015). *WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians*. Computerized Medical Imaging and Graphics, 43, 99–111. Xem thông tin trích dẫn tại <https://polyp.grand-challenge.org/CVCClinicDB/>.
