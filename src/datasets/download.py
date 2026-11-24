"""Pinned-source downloader extracted from downdata.ipynb; import performs no I/O."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import ssl
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from PIL import Image
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

from .common import find_project_root

DATASETS = [{'name': 'Kvasir-SEG',
  'download_type': 'zip',
  'url': 'https://datasets.simula.no/downloads/kvasir-seg.zip',
  'archive_name': 'kvasir-seg.zip',
  'archive_sha256': '03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7',
  'expected_pairs': 1000,
  'extension': '.jpg',
  'image_suffix': ('images',),
  'mask_suffix': ('masks',),
  'expected_size': None,
  'source_page': 'https://datasets.simula.no/kvasir-seg/',
  'role': 'train/validation/test internal; see splits/ for fixed IDs'},
 {'name': 'CVC-ClinicDB',
  'download_type': 'huggingface',
  'repo_id': 'berkaytrhn/cvc-clinicdb',
  'revision': '07f1aae53befabc313f9435c8b1b9565642a36c8',
  'url': 'https://huggingface.co/datasets/berkaytrhn/cvc-clinicdb',
  'expected_pairs': 612,
  'extension': '.png',
  'source_directories': {'images': 'CVC-ClinicDB/Original', 'masks': 'CVC-ClinicDB/Ground Truth'},
  'expected_size': (384, 288),
  'source_page': 'https://polyp.grand-challenge.org/CVCClinicDB/',
  'mirror_page': 'https://huggingface.co/datasets/berkaytrhn/cvc-clinicdb',
  'role': 'external test; không dùng tinh chỉnh mô hình'}]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

class DatasetDownloader:
    def __init__(self, project_root):
        self.project_root = find_project_root(project_root)
        self.data_dir = self.project_root / "data"
        self.download_dir = self.data_dir / "_downloads"

    def sha256_file(self, path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_archive(self, path, expected_sha256):
        actual = self.sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(f'Checksum ZIP không khớp: {path}\nKỳ vọng: {expected_sha256}\nThực tế: {actual}\nKiểm tra nguồn; nếu cache hỏng, di chuyển riêng ZIP này ra ngoài cache rồi chạy lại.')
        if not zipfile.is_zipfile(path):
            raise ValueError(f'File không phải ZIP hợp lệ: {path}')

    def download_archive(self, spec):
        destination = self.download_dir / spec['archive_name']
        if destination.exists():
            self.verify_archive(destination, spec['archive_sha256'])
            print(f'Dùng ZIP cache: {destination.name}')
            return destination
        partial = destination.with_suffix(destination.suffix + '.part')
        for attempt in range(1, 4):
            try:
                request = Request(spec['url'], headers={'User-Agent': 'Medical-Image-Segmentation/1.0'})
                with urlopen(request, timeout=60) as response, partial.open('wb') as handle:
                    total = int(response.headers.get('Content-Length') or 0) or None
                    downloaded = 0
                    with tqdm(total=total, unit='B', unit_scale=True, desc=spec['name']) as progress:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            downloaded += len(chunk)
                            progress.update(len(chunk))
                    if total is not None and downloaded != total:
                        raise OSError(f'Tải thiếu dữ liệu: {downloaded}/{total} byte')
                if not zipfile.is_zipfile(partial):
                    raise ValueError('Máy chủ không trả ZIP (có thể là trang HTML/đăng nhập). Kiểm tra URL và kết nối, không nhập API key vào notebook.')
                self.verify_archive(partial, spec['archive_sha256'])
                partial.replace(destination)
                return destination
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                partial.unlink(missing_ok=True)
                ssl_error = isinstance(exc, ssl.SSLError) or isinstance(getattr(exc, 'reason', None), ssl.SSLError)
                if ssl_error:
                    raise RuntimeError('Không xác minh được chứng chỉ HTTPS. Cập nhật Python/chứng chỉ của môi trường; không dùng verify=False. Chi tiết lỗi gốc ở dưới.') from exc
                if isinstance(exc, ValueError) or (isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}):
                    raise RuntimeError(f"Không tải được {spec['name']}: {exc}") from exc
                if attempt == 3:
                    raise RuntimeError(f"Không tải được {spec['name']} sau 3 lần thử. Kiểm tra mạng và chạy lại.") from exc
                print(f'Lần tải {attempt}/3 thất bại: {exc}. Thử lại...')

    def archive_members(self, archive, spec):
        groups = {'images': {}, 'masks': {}}
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or path.suffix.lower() != spec['extension'] or '__MACOSX' in path.parts:
                continue
            if path.is_absolute() or '..' in path.parts or chr(92) in member.filename or (':' in member.filename):
                raise ValueError(f'Tên thành viên ZIP không an toàn: {member.filename}')
            parent = tuple((part.casefold() for part in path.parts[:-1]))
            for group, suffix in [('images', spec['image_suffix']), ('masks', spec['mask_suffix'])]:
                if parent[-len(suffix):] == suffix:
                    if path.name in groups[group]:
                        raise ValueError(f'Trùng tên trong ZIP: {group}/{path.name}')
                    groups[group][path.name] = member
        images, masks = (groups['images'], groups['masks'])
        expected = spec['expected_pairs']
        if len(images) != expected or len(masks) != expected or set(images) != set(masks):
            raise ValueError(f"{spec['name']}: ZIP phải có {expected} cặp khớp tên; thực tế {len(images)} ảnh, {len(masks)} mask.")
        return groups

    def write_new_file(self, path, payload):
        with path.open('xb') as handle:
            try:
                handle.write(payload)
            except BaseException:
                handle.close()
                path.unlink(missing_ok=True)
                raise

    def prepare_zip_dataset(self, spec):
        zip_path = self.download_archive(spec)
        target = self.data_dir / spec['name']
        with zipfile.ZipFile(zip_path) as archive:
            groups = self.archive_members(archive, spec)
            hashes = {}
            pending = []
            conflicts = []
            for group, members in groups.items():
                directory = target / group
                directory.mkdir(parents=True, exist_ok=True)
                unexpected = [p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and (p.name not in members)]
                if unexpected:
                    conflicts.append(f'{directory}: file ảnh ngoài danh sách nguồn: {unexpected[:5]}')
                for filename, member in tqdm(sorted(members.items()), desc=f"Đối chiếu {spec['name']}/{group}", leave=False):
                    payload = archive.read(member)
                    digest = hashlib.sha256(payload).hexdigest()
                    relative = f'{group}/{filename}'
                    hashes[relative] = digest
                    destination = directory / filename
                    if destination.exists():
                        if not destination.is_file() or self.sha256_file(destination) != digest:
                            conflicts.append(str(destination))
                    else:
                        pending.append((member, destination))
            if conflicts:
                raise RuntimeError('Phát hiện dữ liệu khác nguồn. Không ghi đè/xóa file; kiểm tra các đường dẫn sau:\n' + '\n'.join(conflicts[:10]))
            for member, destination in tqdm(pending, desc=f"Tạo file {spec['name']}", leave=False):
                self.write_new_file(destination, archive.read(member))
        print(f"{spec['name']}: thêm {len(pending)} file, giữ {len(hashes) - len(pending)} file khớp nguồn.")
        return hashes

    def prepare_huggingface_dataset(self, spec):
        api = HfApi(token=False)
        entries = api.list_repo_tree(repo_id=spec['repo_id'], repo_type='dataset', revision=spec['revision'], path_in_repo='CVC-ClinicDB', recursive=True)
        sources = {}
        expected_names = {f'{index}.png' for index in range(1, spec['expected_pairs'] + 1)}
        source_groups = {directory: group for group, directory in spec['source_directories'].items()}
        for entry in entries:
            path = PurePosixPath(entry.path)
            if str(path.parent) not in source_groups or path.suffix.lower() != '.png':
                continue
            if path.is_absolute() or '..' in path.parts or chr(92) in entry.path or (':' in entry.path):
                raise ValueError(f'Đường dẫn Hugging Face không an toàn: {entry.path}')
            lfs = getattr(entry, 'lfs', None)
            digest = getattr(lfs, 'sha256', None)
            if not digest or len(digest) != 64:
                raise ValueError(f'Thiếu SHA-256 LFS tại revision đã chốt: {entry.path}')
            relative = f'{source_groups[str(path.parent)]}/{path.name}'
            if relative in sources:
                raise ValueError(f'Trùng file trong nguồn Hugging Face: {relative}')
            sources[relative] = (entry.path, digest)
        for group in ('images', 'masks'):
            names = {relative.split('/', 1)[1] for relative in sources if relative.startswith(group + '/')}
            if names != expected_names:
                raise ValueError(f"{spec['name']}: nguồn Hugging Face phải có đủ 612 tên PNG cho {group}.")
        target = self.data_dir / spec['name']
        conflicts = []
        pending = []
        for group in ('images', 'masks'):
            directory = target / group
            directory.mkdir(parents=True, exist_ok=True)
            unexpected = [p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and (p.name not in expected_names)]
            if unexpected:
                conflicts.append(f'{directory}: file ảnh ngoài danh sách nguồn: {unexpected[:5]}')
        for relative, (remote_path, digest) in tqdm(sorted(sources.items()), desc='Đối chiếu CVC-ClinicDB/Hugging Face', leave=False):
            destination = target / relative
            if destination.exists():
                if not destination.is_file() or self.sha256_file(destination) != digest:
                    conflicts.append(str(destination))
            else:
                pending.append((relative, remote_path, digest))
        if conflicts:
            raise RuntimeError('Phát hiện dữ liệu khác revision Hugging Face. Không ghi đè/xóa file:\n' + '\n'.join(conflicts[:10]))
        if pending:
            cache_dir = self.download_dir / 'clinicdb-hf'
            snapshot_dir = Path(snapshot_download(repo_id=spec['repo_id'], repo_type='dataset', revision=spec['revision'], local_dir=str(cache_dir), token=False, max_workers=4, allow_patterns=[remote_path for _, remote_path, _ in pending] + ['metadata.csv', 'class_dict.csv', 'README.md']))
            for relative, remote_path, digest in pending:
                source = snapshot_dir / remote_path
                if not source.is_file() or self.sha256_file(source) != digest:
                    raise ValueError(f'File cache Hugging Face sai SHA-256: {source}. Kiểm tra/di chuyển riêng file hỏng rồi chạy lại.')
            for relative, remote_path, digest in tqdm(pending, desc='Tạo file CVC-ClinicDB', leave=False):
                self.write_new_file(target / relative, (snapshot_dir / remote_path).read_bytes())
        print(f'CVC-ClinicDB: thêm {len(pending)} file, giữ {len(sources) - len(pending)} file khớp revision Hugging Face.')
        return {relative: digest for relative, (_, digest) in sources.items()}

    def prepare_dataset(self, spec):
        if spec['download_type'] == 'zip':
            return self.prepare_zip_dataset(spec)
        if spec['download_type'] == 'huggingface':
            return self.prepare_huggingface_dataset(spec)
        raise ValueError(f"Nguồn tải chưa hỗ trợ: {spec['download_type']}")

    def validate_dataset(self, spec, hashes):
        folder = self.data_dir / spec['name']
        names = sorted((key.split('/', 1)[1] for key in hashes if key.startswith('images/')))
        expected = spec['expected_pairs']
        for group in ('images', 'masks'):
            files = [p.name for p in (folder / group).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            if len(files) != expected or set(files) != set(names):
                raise ValueError(f"{spec['name']}/{group}: không đủ {expected} file khớp nguồn.")
        sizes = Counter()
        non_binary_masks = 0
        expected_format = 'JPEG' if spec['extension'] == '.jpg' else 'PNG'
        for filename in tqdm(names, desc=f"Kiểm tra {spec['name']}", leave=False):
            image_path, mask_path = (folder / 'images' / filename, folder / 'masks' / filename)
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image.load()
                mask.load()
                if image.format != expected_format or mask.format != expected_format:
                    raise ValueError(f'Định dạng không khớp phần mở rộng: {filename}')
                if image.size != mask.size:
                    raise ValueError(f'Kích thước ảnh–mask không khớp: {filename}')
                if spec['expected_size'] and image.size != spec['expected_size']:
                    raise ValueError(f'CVC-ClinicDB sai kích thước: {filename}, {image.size}')
                sizes[f'{image.width}x{image.height}'] += 1
                histogram = mask.convert('L').histogram()
                foreground = sum(histogram[128:])
                if foreground == 0 or foreground == image.width * image.height:
                    raise ValueError(f'Mask không có đủ nền và polyp: {filename}')
                non_binary_masks += int(any(histogram[1:255]))
        return {'pairs': len(names), 'extension': spec['extension'], 'size_counts': dict(sorted(sizes.items())), 'masks_with_intermediate_gray_values': non_binary_masks}

    def write_manifest(self, path, manifest):
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, suffix='.tmp', delete=False) as handle:
                temporary_path = Path(handle.name)
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    def run(self):
        for folder in (self.data_dir, self.download_dir, self.project_root / "logs", self.project_root / "checkpoints"):
            folder.mkdir(parents=True, exist_ok=True)
        datasets = [dict(spec) for spec in DATASETS]
        datasets[0]["role"] = "train/validation/test nội bộ; xem splits/ để biết danh sách đã tạo"
        manifest = {
            'schema_version': 2,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'mask_check_threshold': 127,
            'datasets': {},
        }
        summaries = {}
        for spec in datasets:
            file_hashes = self.prepare_dataset(spec)
            summary = self.validate_dataset(spec, file_hashes)
            summaries[spec['name']] = summary
            manifest['datasets'][spec['name']] = {
                'download_url': spec['url'],
                'download_type': spec['download_type'],
                'source_page': spec['source_page'],
                'mirror_page': spec.get('mirror_page'),
                'archive_sha256': spec.get('archive_sha256'),
                'repo_id': spec.get('repo_id'),
                'revision': spec.get('revision'),
                'role': spec['role'],
                'summary': summary,
                'files_sha256': file_hashes,
            }

        manifest_path = self.data_dir / 'dataset_manifest.json'
        self.write_manifest(manifest_path, manifest)
        print('\nHoàn tất. Manifest:', manifest_path)
        print(f"{'Dataset':<18} {'Ảnh':>6} {'Mask':>6} {'Định dạng':>10}")
        for name, summary in summaries.items():
            print(f"{name:<18} {summary['pairs']:>6} {summary['pairs']:>6} {summary['extension']:>10}")
            print('  Mask có mức xám trung gian:', summary['masks_with_intermediate_gray_values'])
        print('Dữ liệu sẵn sàng. Chưa tạo split và chưa huấn luyện mô hình.')
        return manifest


