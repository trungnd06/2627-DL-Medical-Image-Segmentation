"""Compare completed validation runs without loading models or test data."""
import csv
import json
from pathlib import Path
from .utils import load_config
from statistics import mean, median


def load_run_summary(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('mode') != 'full' or metadata.get('status') != 'complete':
        raise ValueError(f'Expected a completed full run: {run_dir}')
    with (run_dir / 'best_val_metrics.csv').open(encoding='utf-8-sig') as source:
        rows = list(csv.DictReader(source))
    scores = {row['image_id']: float(row['dice']) for row in rows}
    if len(scores) != len(rows) or not rows:
        raise ValueError('Per-image metrics must be nonempty with unique image IDs')
    with (run_dir / 'history.csv').open(encoding='utf-8-sig') as source:
        history = list(csv.DictReader(source))
    best = next(row for row in history if int(row['epoch']) == metadata['best_epoch'])
    return {'metadata': metadata, 'config': load_config(run_dir / 'config.yaml'),
            'scores': scores, 'history': history, 'best': best,
            'dice': mean(scores.values()), 'iou': mean(float(row['iou']) for row in rows),
            'median': median(scores.values()), 'low': sum(v < .5 for v in scores.values()),
            'high': sum(v >= .9 for v in scores.values())}


def ablation_table(a1_dir, m2_dir):
    """Paired validation report from recorded checkpoints, with control checks."""
    a1, m2 = load_run_summary(a1_dir), load_run_summary(m2_dir)
    if (a1['metadata']['model_provenance']['encoder_weights'] is not None
            or m2['metadata']['model_provenance']['encoder_weights'] != 'imagenet'):
        raise ValueError('Select A1 with random encoder and M2 with ImageNet encoder')
    if a1['scores'].keys() != m2['scores'].keys():
        raise ValueError('A1/M2 validation image IDs differ')
    rows = [('| Chỉ tiêu | A1 — encoder ngẫu nhiên | M2 — encoder ImageNet |'),
            '| --- | ---: | ---: |']
    for label, values in (
        ('Validation Dice', [f"{s['dice']:.4f}" for s in (a1, m2)]),
        ('Validation IoU', [f"{s['iou']:.4f}" for s in (a1, m2)]),
        ('Trung vị Dice', [f"{s['median']:.4f}" for s in (a1, m2)]),
        ('Số ảnh Dice ≥0.9', [str(s['high']) for s in (a1, m2)]),
        ('Số ảnh Dice <0.5', [str(s['low']) for s in (a1, m2)]),
        ('Số tham số', [f"{s['metadata']['parameters']:,}" for s in (a1, m2)]),
        ('Epoch được chọn', [str(s['metadata']['best_epoch']) for s in (a1, m2)]),
        ('Số epoch thực thi', [str(s['metadata']['epochs_completed']) for s in (a1, m2)]),
        ('Thời gian ghi nhận (phút)', [f"{s['metadata']['elapsed_seconds']/60:.2f}" for s in (a1, m2)])):
        rows.append(f'| {label} | {values[0]} | {values[1]} |')
    controls = a1['metadata']['split_sha256'] == m2['metadata']['split_sha256']
    controls &= all(a1['config'][key] == m2['config'][key]
                    for key in ('seed', 'data', 'loader', 'training', 'evaluation'))
    models = [{k: v for k, v in s['config']['model'].items() if k != 'encoder_weights'} for s in (a1, m2)]
    controls &= models[0] == models[1]
    provenance = [s['metadata']['model_provenance'] for s in (a1, m2)]
    same_head = all(provenance[0][key] == provenance[1][key]
                    for key in ('decoder_initial_sha256', 'head_initial_sha256'))
    rows += [f'\nĐiều kiện cấu hình và split trùng khớp: **{bool(controls)}**.',
             f'\nKhởi tạo decoder và lớp đầu ra trùng khớp: **{same_head}**.',
             f"\nChênh lệch M2 − A1: **{100*(m2['dice']-a1['dice']):.2f} điểm phần trăm Dice**, **{100*(m2['iou']-a1['iou']):.2f} điểm phần trăm IoU**."]
    improved = sum(m2['scores'][key] > a1['scores'][key] for key in a1['scores'])
    worse = sum(m2['scores'][key] < a1['scores'][key] for key in a1['scores'])
    rows += [f'\nĐối chiếu từng ảnh: M2 tốt hơn ở **{improved}** ảnh, thấp hơn ở **{worse}** ảnh.',
             '\nĐây là kết quả validation của từng checkpoint được chọn, không phải đánh giá test hoặc kiểm định ý nghĩa thống kê.']
    if not all(s['metadata']['review_status']['ready_for_final_reporting'] for s in (a1,m2)):
        rows.append('\nTrạng thái rà dữ liệu: **provisional**; cần giữ giới hạn này khi diễn giải kết quả.')
    result = '\n'.join(rows)
    (Path(a1_dir) / 'A1_M2_validation_comparison.md').write_text(result+'\n', encoding='utf-8')
    return result


def compare_saved_runs(project_root):
    root = Path(project_root)
    selected = []
    for experiment in ('M1', 'M2', 'A1'):
        candidates = []
        for path in (root / 'results' / experiment).glob('*/metadata.json'):
            meta = json.loads(path.read_text(encoding='utf-8'))
            if meta.get('mode') == 'full' and meta.get('status') == 'complete':
                candidates.append((path, meta))
        if candidates:
            path, meta = max(candidates, key=lambda item: item[0].stat().st_mtime)
            config = load_config(path.parent / 'config.yaml')
            with (path.parent / 'best_val_metrics.csv').open(encoding='utf-8-sig') as source:
                rows = list(csv.DictReader(source))
            signature = {'splits': meta['split_sha256'], 'data':config['data'], 'loader':config['loader'],
                         'seed':config['seed'], 'training':config['training'], 'evaluation':config['evaluation']}
            selected.append((experiment, meta, signature, rows))
    lines = ['| Model | Run | Best epoch | Val Dice | Val IoU | Parameters | Seconds | Review |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for experiment, meta, _, rows in selected:
        iou = sum(float(row['iou']) for row in rows) / len(rows)
        lines.append(f"| {experiment} | {meta['run_id']} | {meta['best_epoch']} | {meta['best_dice']:.4f} | {iou:.4f} | {meta['parameters']} | {meta['elapsed_seconds']:.1f} | {'checked' if meta['review_status']['ready_for_final_reporting'] else 'provisional'} |")
    if not selected:
        lines.append('\nNo completed full runs found.')
    elif any(item[2] != selected[0][2] for item in selected[1:]):
        lines.append('\nWARNING: split/config/seed/device settings differ; inspect saved configs before drawing controlled-comparison conclusions.')
    paired = {name: meta.get('model_provenance') for name, meta, _, _ in selected}
    if paired.get('M2') and paired.get('A1'):
        matched = all(paired['M2'][key] == paired['A1'][key] for key in ('decoder_initial_sha256','head_initial_sha256'))
        lines.append(f'\nA1/M2 decoder and head initialization match: {matched}.')
    lines.append('\nValidation only; no test/external inference. M1/M2 differ in architecture and pretraining.')
    result = '\n'.join(lines)
    output = root / 'results/tables/M1_M2_A1_validation.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result+'\n', encoding='utf-8')
    return result
