"""Download VGG16 ImageNet weights into weights/ for offline/air-gapped training."""

import sys
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
FILENAME = "vgg16-397923af.pth"
URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


def main() -> None:
    dest = WEIGHTS_DIR / FILENAME
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Already exists: {dest}")
        print("Delete the file and re-run to force a fresh download.")
        return

    try:
        import torch
    except ImportError:
        print("torch is required. Run: pip install torch", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading VGG16 weights (~528 MB) → {dest}")
    torch.hub.download_url_to_file(URL, str(dest), progress=True)
    print(f"\nSaved: {dest}")
    print("Point your config at it: " "training.vgg_weights_path=weights/vgg16-397923af.pth")


if __name__ == "__main__":
    main()
