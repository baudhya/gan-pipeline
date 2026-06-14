#!/usr/bin/env bash
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[init]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
die()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── 1. Python version check ───────────────────────────────────────────────────
PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" &>/dev/null; then
    die "Python not found. Install Python ≥ 3.10 and re-run."
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 || ("$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10) ]]; then
    die "Python 3.10+ required (found $PY_VERSION). Set PYTHON= to point at a newer interpreter."
fi

info "Python $PY_VERSION ✓"

# ── 2. Virtual environment ────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-.venv}"

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation."
else
    info "Creating virtual environment at $VENV_DIR …"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Virtual environment activated."

# ── 3. Upgrade pip ───────────────────────────────────────────────────────────
info "Upgrading pip …"
pip install --quiet --upgrade pip

# ── 4. Install dependencies ───────────────────────────────────────────────────
info "Installing core + dev dependencies …"
pip install --quiet -e ".[dev]"

# Optional extras
if [[ "${INSTALL_GEO:-0}" == "1" ]]; then
    info "Installing geospatial extras (rasterio, h5py) …"
    pip install --quiet -e ".[geo]"
fi

if [[ "${INSTALL_EVAL:-0}" == "1" ]]; then
    info "Installing evaluation extras (torch-fidelity) …"
    pip install --quiet -e ".[eval]"
fi

# ── 5. Pre-commit hooks ───────────────────────────────────────────────────────
info "Installing pre-commit hooks …"
pre-commit install

# ── 6. VGG weights ───────────────────────────────────────────────────────────
VGG_WEIGHTS="weights/vgg16-397923af.pth"
if [[ -f "$VGG_WEIGHTS" ]]; then
    info "VGG16 weights already present — skipping download."
elif [[ "${DOWNLOAD_VGG:-0}" == "1" ]]; then
    info "Downloading VGG16 weights (~528 MB) …"
    python scripts/download_vgg_weights.py
else
    warn "VGG16 weights not found at $VGG_WEIGHTS."
    warn "Run  make download-weights  before training with VGG loss."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Setup complete. Activate your environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
info "Quick-start:"
echo "    make download-weights   # fetch VGG16 weights (once)"
echo "    make prepare-data       # convert Sentinel data to PNGs"
echo "    make train              # start training"
