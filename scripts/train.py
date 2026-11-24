"""Chạy M1 từ terminal; logic nằm trong src.training."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import run_training


def main():
    parser = argparse.ArgumentParser(description='M1 U-Net: smoke, overfit or full training')
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--config', default='configs/unet.yaml')
    parser.add_argument('--mode', choices=('smoke', 'overfit', 'full'), default='overfit')
    parser.add_argument('--epochs', type=int, help='Override max_epochs; smoke defaults to 1')
    parser.add_argument('--samples', type=int, default=8, help='Training samples for smoke/overfit; ignored in full mode')
    parser.add_argument('--device', default=None, help='auto, cpu, cuda or cuda:N')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--quiet', action='store_true', help='Hide batch progress bars')
    args = parser.parse_args()
    run_training(args.project_root, args.config, mode=args.mode, epochs=args.epochs,
                 samples=args.samples, device=args.device, batch_size=args.batch_size, progress=not args.quiet)


if __name__ == '__main__':
    main()
