#!/usr/bin/env bash
# Reproducible CPU-first bootstrap for ENVEDA-CASMI26.
# Usage: bash setup.sh [--core | --full | --no-download]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:---core}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Install only missing OS utilities; do not require sudo if the machine is already prepared.
if command -v apt-get >/dev/null 2>&1; then
  MISSING=()
  for tool in git curl wget unzip; do command -v "$tool" >/dev/null 2>&1 || MISSING+=("$tool"); done
  if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1 2; then MISSING+=(python3-venv); fi
  if ((${#MISSING[@]})); then
    SUDO=()
    if [[ "$EUID" -ne 0 ]]; then
      if command -v sudo >/dev/null 2>&1; then SUDO=(sudo); else echo "Missing system tools (${MISSING[*]}) and sudo is unavailable." >&2; exit 2; fi
    fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y git curl wget unzip build-essential python3-venv
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.10+ (3.11 recommended)." >&2; exit 2
fi
PYVER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYVER" in 3.10|3.11|3.12|3.13) ;; *) echo "WARNING: Python $PYVER; Python 3.11 is the tested target.";; esac

if [[ ! -d .venv ]]; then "$PYTHON_BIN" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# Explicit CPU-only PyTorch wheel. No CUDA package is pulled on a CPU machine.
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu

mkdir -p data/{competition,benchmark,coconut,gnps,massbank} repos models outputs cache logs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

python - <<'PY'
import torch, numpy, pandas, optuna
from rdkit import Chem
print('Python packages OK')
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('numpy:', numpy.__version__, '| pandas:', pandas.__version__, '| optuna:', optuna.__version__)
print('RDKit:', Chem.rdBase.rdkitVersion)
PY

case "$MODE" in
  --no-download)
    echo "Environment ready. Skipping repository and data downloads." ;;
  --core)
    python scripts/bootstrap_assets.py --repos --competition
    echo "Core setup complete. Run: source .venv/bin/activate" ;;
  --full)
    python scripts/bootstrap_assets.py --all
    if [[ -s data/gnps/GNPS-LIBRARY.mgf ]]; then
      python scripts/prepare_gnps.py --max-spectra "${GNPS_MAX_SPECTRA:-250000}"
    fi
    echo "Full setup attempted. Some assets may require Kaggle credentials/license acceptance." ;;
  *) echo "Usage: bash setup.sh [--core | --full | --no-download]" >&2; exit 2 ;;
esac

echo "Activate with: source \"$ROOT/.venv/bin/activate\""
echo "Inspect downloaded schemas: python scripts/inspect_data.py"
echo "Run 10-hour CPU NAS: python run_nas.py --profile cpu_10h"
