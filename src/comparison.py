"""Compare completed validation runs without loading models or test data."""
import csv
import json
from pathlib import Path
from .utils import load_config


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
