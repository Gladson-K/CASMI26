"""Download competition, COCONUT, and benchmark datasets."""
import os
import subprocess
import zipfile
from pathlib import Path

def _run(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def download_competition(cfg):
    out = Path(cfg['paths']['competition_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.parquet").exists():
        print("[skip] competition data present")
        return
    try:
        _run(f"kaggle competitions download -c {cfg['download']['competition']} -p {out}")
        zip_path = out / f"{cfg['download']['competition']}.zip"
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out)
        zip_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[warn] competition download failed: {e}")

def download_coconut(cfg):
    out = Path(cfg['paths']['coconut_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.sdf")):
        print("[skip] COCONUT data present")
        return
    zip_path = out / "coconut.zip"
    _run(f"wget -q --show-progress -O {zip_path} {cfg['download']['coconut_url']}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out)
    zip_path.unlink(missing_ok=True)

def download_benchmark(cfg):
    out = Path(cfg['paths']['benchmark_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.parquet")):
        print("[skip] benchmark data present")
        return
    try:
        _run(f"kaggle datasets download -d {cfg['download']['benchmark_dataset']} -p {out}")
        for zp in out.glob("*.zip"):
            with zipfile.ZipFile(zp, 'r') as z:
                z.extractall(out)
            zp.unlink(missing_ok=True)
    except Exception as e:
        print(f"[warn] benchmark download failed: {e}")

def download_all(cfg):
    download_competition(cfg)
    download_coconut(cfg)
    download_benchmark(cfg)