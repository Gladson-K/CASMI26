#!/usr/bin/env python3
"""Idempotent asset bootstrap. Failed optional sources are recorded, not hidden."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs' / 'cpu_10h.yaml'
try:
    import yaml
    config = yaml.safe_load(CFG.read_text())
except Exception:
    config = {}
PATHS = config.get('paths', {})
DOWNLOAD = config.get('download', {})
REPOS = {
    'DreaMS': DOWNLOAD.get('dreams_repo', 'https://github.com/pluskal-lab/DreaMS.git'),
    'spec2vec': DOWNLOAD.get('spec2vec_repo', 'https://github.com/iomega/spec2vec.git'),
    'matchms': DOWNLOAD.get('matchms_repo', 'https://github.com/matchms/matchms.git'),
    'MSBuddy': DOWNLOAD.get('msbuddy_repo', 'https://github.com/CompMS-MS2/MSBuddy.git'),
}
REPORT = ROOT / 'outputs' / 'asset_download_report.json'


def run(args, *, cwd=None, check=True):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=check)


def safe_extract(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f'Unsafe path in archive: {member.filename}')
        zf.extractall(dest)


def download_url(url: str, target: Path, *, min_bytes=1000):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size >= min_bytes:
        print(f'[skip] {target} ({target.stat().st_size:,} bytes)')
        return
    tmp = target.with_suffix(target.suffix + '.part')
    print(f'[download/resume] {url} -> {target}', flush=True)
    # curl can resume large MGF/archive downloads if the previous run was interrupted.
    run(['curl', '-fL', '--retry', '3', '-C', '-', '-o', tmp, url])
    if not tmp.exists() or tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'Download too small; endpoint may have returned an error page: {url}')
    tmp.replace(target)


def clone_repos():
    out = ROOT / 'repos'; out.mkdir(exist_ok=True)
    report = {}
    for name, url in REPOS.items():
        dest = out / name
        try:
            if (dest / '.git').exists():
                run(['git', '-C', dest, 'fetch', '--depth', '1', 'origin'], check=False)
                run(['git', '-C', dest, 'pull', '--ff-only'], check=False)
                status = 'updated-or-existing'
            elif dest.exists() and any(dest.iterdir()):
                status = 'skipped: destination exists but is not a git checkout'
            else:
                run(['git', 'clone', '--depth', '1', url, dest])
                status = 'cloned'
            revision = None
            if (dest / '.git').exists():
                rev = subprocess.run(['git', '-C', str(dest), 'rev-parse', 'HEAD'], capture_output=True, text=True)
                if rev.returncode == 0: revision = rev.stdout.strip()
            report[name] = {'url': url, 'path': str(dest.relative_to(ROOT)), 'status': status, 'revision': revision}
        except Exception as exc:
            report[name] = {'url': url, 'path': str(dest.relative_to(ROOT)), 'status': 'failed', 'error': str(exc)}
            print(f'[warn] repo {name}: {exc}', file=sys.stderr)
    return report


def kaggle_ready():
    try:
        p = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        return p.returncode == 0
    except FileNotFoundError:
        return False


def download_competition():
    out = ROOT / PATHS.get('competition_dir', 'data/competition'); out.mkdir(parents=True, exist_ok=True)
    if (out / 'train.parquet').exists() and (out / 'test.parquet').exists():
        return {'status': 'already-present', 'path': str(out.relative_to(ROOT))}
    if not kaggle_ready():
        return {'status': 'skipped', 'reason': 'Kaggle CLI unavailable; activate .venv and configure ~/.kaggle/kaggle.json'}
    slug = DOWNLOAD.get('competition', 'enveda-CASMI26-molecule-id-mass-spectra')
    try:
        run(['kaggle', 'competitions', 'download', slug, '-p', out])
        archives = list(out.glob('*.zip'))
        if not archives:
            return {'status': 'failed', 'reason': 'Kaggle command completed but no ZIP was found; inspect output'}
        for archive in archives:
            safe_extract(archive, out)
            archive.unlink(missing_ok=True)
        return {'status': 'downloaded', 'path': str(out.relative_to(ROOT)), 'files': sorted(p.name for p in out.iterdir())}
    except Exception as exc:
        return {'status': 'failed', 'reason': str(exc), 'hint': 'Accept the competition rules on Kaggle and check API credentials.'}


def download_benchmark():
    out = ROOT / PATHS.get('benchmark_dir', 'data/benchmark'); out.mkdir(parents=True, exist_ok=True)
    if any(out.rglob('*.parquet')):
        return {'status': 'already-present', 'path': str(out.relative_to(ROOT))}
    if not kaggle_ready():
        return {'status': 'skipped', 'reason': 'Kaggle CLI unavailable'}
    slug = DOWNLOAD.get('benchmark_dataset', 'dmitriigluzdov/casmi26-natural-product-panel-benchmark')
    try:
        run(['kaggle', 'datasets', 'download', slug, '-p', out])
        for archive in list(out.glob('*.zip')):
            safe_extract(archive, out); archive.unlink(missing_ok=True)
        return {'status': 'downloaded', 'path': str(out.relative_to(ROOT))}
    except Exception as exc:
        return {'status': 'failed', 'reason': str(exc), 'hint': 'The referenced dataset slug may be private, removed, or renamed.'}


def download_coconut():
    out = ROOT / PATHS.get('coconut_dir', 'data/coconut'); out.mkdir(parents=True, exist_ok=True)
    if any(out.rglob('*.sdf')) or any(out.rglob('*.sdf.gz')):
        return {'status': 'already-present', 'path': str(out.relative_to(ROOT))}
    url = DOWNLOAD.get('coconut_url', '')
    try:
        archive = out / 'coconut_download.zip'
        download_url(url, archive, min_bytes=10000)
        safe_extract(archive, out); archive.unlink(missing_ok=True)
        return {'status': 'downloaded', 'path': str(out.relative_to(ROOT))}
    except Exception as exc:
        return {'status': 'failed', 'reason': str(exc), 'hint': 'COCONUT endpoint/version may have changed; use the official download page and retain its license.'}


def download_gnps():
    out = ROOT / PATHS.get('gnps_dir', 'data/gnps'); out.mkdir(parents=True, exist_ok=True)
    target = out / 'GNPS-LIBRARY.mgf'
    try:
        download_url(DOWNLOAD.get('gnps_mgf_url', 'https://external.gnps2.org/gnpslibrary/GNPS-LIBRARY.mgf'), target, min_bytes=100000)
        return {'status': 'downloaded-or-present', 'path': str(target.relative_to(ROOT)), 'bytes': target.stat().st_size,
                'license_note': 'Check licenses per included library; GNPS aggregate can include third-party libraries.'}
    except Exception as exc:
        return {'status': 'failed', 'reason': str(exc), 'hint': 'GNPS is a large download; see https://external.gnps2.org/gnpslibrary and choose MGF/MSP.'}


def download_dreams_weights():
    """Download official pretrained DreaMS checkpoints from Zenodo with MD5 checks."""
    import hashlib
    import requests
    out = ROOT / 'models' / 'dreams'; out.mkdir(parents=True, exist_ok=True)
    assets = {
        'embedding_model.ckpt': '4f6913c242b219b75e667dd1ed5add59',
        'ssl_model.ckpt': '4deee91dd4486c2b6cbcde1f200b8f50',
    }
    report = {}
    try:
        record = requests.get('https://zenodo.org/api/records/10997887', timeout=45)
        record.raise_for_status()
        file_meta = {f['key']: f for f in record.json().get('files', [])}
    except Exception as exc:
        return {'status': 'failed', 'reason': f'Cannot read Zenodo record metadata: {exc}'}
    for name, expected_md5 in assets.items():
        target = out / name
        meta = file_meta.get(name)
        if not meta:
            report[name] = {'status': 'failed', 'reason': 'File not found in Zenodo record metadata'}
            continue
        expected_size = int(meta.get('size', 0))
        url = meta.get('links', {}).get('self') or f'https://zenodo.org/records/10997887/files/{name}?download=1'
        try:
            valid = False
            if target.exists() and target.stat().st_size == expected_size:
                h = hashlib.md5()
                with target.open('rb') as fh:
                    for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''): h.update(chunk)
                valid = h.hexdigest() == expected_md5
            if not valid:
                if target.exists() and target.stat().st_size >= expected_size:
                    target.unlink()
                # Keep a partial file so curl can resume interrupted multi-GB downloads.
                run(['curl', '-fL', '--retry', '3', '-C', '-', '-o', target, url])
                if not target.exists() or target.stat().st_size != expected_size:
                    raise RuntimeError(f'Unexpected size for {name}: {target.stat().st_size if target.exists() else 0} != {expected_size}')
                h = hashlib.md5()
                with target.open('rb') as fh:
                    for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''): h.update(chunk)
                if h.hexdigest() != expected_md5:
                    raise RuntimeError(f'MD5 mismatch for {name}: got {h.hexdigest()}')
            report[name] = {'status': 'verified', 'path': str(target.relative_to(ROOT)), 'md5': expected_md5, 'bytes': target.stat().st_size}
        except Exception as exc:
            report[name] = {'status': 'failed', 'reason': str(exc), 'url': url}
            print(f'[warn] DreaMS checkpoint {name}: {exc}', file=sys.stderr)
    report['source'] = 'https://zenodo.org/records/10997887'
    report['note'] = 'Both checkpoints total about 2.6 GB. Downloading files does not install DreaMS or prove CPU inference speed.'
    return report


def download_kaggle_buddy():
    out = ROOT / 'models' / 'kaggle_buddy'
    if out.exists() and any(out.iterdir()):
        return {'status': 'already-present', 'path': str(out.relative_to(ROOT))}
    try:
        import kagglehub
        out.mkdir(parents=True, exist_ok=True)
        path = kagglehub.model_download('jek1wantaufik/buddy/other/enveda', output_dir=str(out))
        return {'status': 'downloaded', 'path': str(path)}
    except Exception as exc:
        return {'status': 'failed-or-not-supported-by-kagglehub', 'reason': str(exc),
                'hint': 'Download the model from its Kaggle page after signing in: https://www.kaggle.com/models/jek1wantaufik/buddy/Other/enveda'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos', action='store_true')
    parser.add_argument('--competition', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--coconut', action='store_true')
    parser.add_argument('--gnps', action='store_true')
    parser.add_argument('--buddy-model', action='store_true')
    parser.add_argument('--dreams-weights', action='store_true')
    parser.add_argument('--all', action='store_true', help='Attempt all downloads; some require authentication or may be large')
    args = parser.parse_args()
    if args.all:
        args.repos = args.competition = args.benchmark = args.coconut = args.gnps = args.buddy_model = args.dreams_weights = True
    if not any(vars(args).values()):
        parser.error('choose --repos, --competition, --benchmark, --coconut, --gnps, --buddy-model, or --all')
    report = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'assets': {}}
    if args.repos: report['assets']['repositories'] = clone_repos()
    if args.competition: report['assets']['competition'] = download_competition()
    if args.benchmark: report['assets']['benchmark'] = download_benchmark()
    if args.coconut: report['assets']['coconut'] = download_coconut()
    if args.gnps: report['assets']['gnps'] = download_gnps()
    if args.buddy_model: report['assets']['buddy_model'] = download_kaggle_buddy()
    if args.dreams_weights: report['assets']['dreams_weights'] = download_dreams_weights()
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(json.dumps(report, indent=2))
    print('\nAsset report:', REPORT)
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
