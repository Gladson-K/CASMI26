#!/usr/bin/env python3
"""Run resumable, CPU-only Optuna search (default 10-hour budget)."""
import argparse, os
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/cpu_10h.yaml')
    p.add_argument('--profile', choices=['cpu_10h'], default=None)
    p.add_argument('--timeout-hours', type=float, default=None)
    p.add_argument('--trials', type=int, default=None)
    p.add_argument('--max-rows', type=int, default=0, help='Optional quick-test cap on parquet rows; 0=all rows')
    p.add_argument('--smoke-test', action='store_true', help='Use <=1000 rows and <=60 seconds')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ('cache_dir', 'output_dir'):
        Path(cfg['paths'][key]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS', str(cfg['project'].get('num_threads', 4)))
    os.environ.setdefault('MKL_NUM_THREADS', str(cfg['project'].get('num_threads', 4)))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(cfg['project'].get('num_threads', 4)))
    from threadpoolctl import threadpool_limits
    from src.nas.runner import load_rows, run_retrieval_nas
    data = Path(cfg['paths']['competition_dir']) / 'train.parquet'
    if not data.exists():
        raise SystemExit(f'Missing {data}. Run bash setup.sh --core, configure Kaggle credentials, then retry.')
    if args.smoke_test:
        args.max_rows = args.max_rows or 1000
        timeout_seconds, trials = 60, 3
    else:
        timeout_seconds = int((args.timeout_hours if args.timeout_hours is not None else cfg['nas'].get('timeout_seconds', 36000) / 3600) * 3600)
        trials = args.trials if args.trials is not None else int(cfg['nas'].get('n_trials', 100000))
    print(f'Loading training data from {data}; max_rows={args.max_rows or "all"}', flush=True)
    df = load_rows(data, max_rows=args.max_rows)
    print(f'Usable spectra: {len(df):,}; unique labels: {df["_label"].nunique():,}', flush=True)
    print(f'Optuna CPU budget: {timeout_seconds / 3600:.2f} hours; max trials={trials}; n_jobs=1 (memory-safe, deterministic).', flush=True)
    with threadpool_limits(limits=int(cfg['project'].get('num_threads', 4))):
        run_retrieval_nas(df, cfg, timeout_seconds=timeout_seconds, n_trials=trials)

if __name__ == '__main__': main()
