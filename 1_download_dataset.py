"""
Step 1 — Download & decompress the AERO-DETR dataset and model checkpoints.

Source (single Zenodo record, v3):
    AERO-DETR Runway Dataset
    DOI: https://doi.org/10.5281/zenodo.21094439

This script fetches two archives from the record and unpacks them into the
repo-local layout that every downstream script/notebook expects:

    rasters/<category>/*.jp2     <- georeferenced imagery (from AERO_DETR_Dataset_DATA.zip)
    models/*.pt                  <- frozen checkpoints    (from models.zip)

Ground-truth vectors (ground_truth/*.geojson) ship inside this repository and
are NOT downloaded here.

Usage
-----
    python 1_download_dataset.py                 # rasters + models into ./
    python 1_download_dataset.py --dest .        # explicit destination
    python 1_download_dataset.py --skip-models   # imagery only
    python 1_download_dataset.py --skip-rasters  # checkpoints only
    python 1_download_dataset.py --keep-zip      # do not delete the .zip after extract

Cross-platform: pure standard library (urllib + zipfile). Works on Windows,
Linux, macOS, and Colab.
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Zenodo record (v3) — https://doi.org/10.5281/zenodo.21094439
# ─────────────────────────────────────────────────────────────
ZENODO_RECORD = "21094439"
ZENODO_DOI = "10.5281/zenodo.21094439"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

DATA_ZIP = "AERO_DETR_Dataset_DATA.zip"   # ~5 GB — imagery + ML annotations
MODELS_ZIP = "models.zip"                 # ~253 MB — frozen checkpoints

# Prefix of the GIS rasters inside DATA_ZIP; everything under here is flattened
# into ./rasters/<category>/...
RASTERS_PREFIX = "AERO_DETR_Dataset_DATA/AERO_DETR_Dataset_GIS/Rasters/"

EXPECTED_MODELS = ("rwy_obb_v1.pt", "rwy_markings_H_v1.pt")


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest` with a simple progress readout."""
    if dest.exists():
        print(f"  Already downloaded: {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"  Downloading {dest.name} from Zenodo ...")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r    {pct:5.1f}%  ({downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB)",
              end="", flush=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    tmp.replace(dest)
    print(f"\n  Saved: {dest}")


def download_rasters(dest: Path, keep_zip: bool) -> None:
    """Download DATA_ZIP and extract only the GIS rasters into ./rasters/<cat>/."""
    rasters_root = dest / "rasters"
    if rasters_root.exists() and any(rasters_root.rglob("*.jp2")):
        print(f"Rasters already present at: {rasters_root}")
        return

    zip_path = dest / DATA_ZIP
    _download(f"{BASE_URL}/{DATA_ZIP}", zip_path)

    print("  Extracting GIS rasters (this may take several minutes) ...")
    rasters_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist()
                   if m.startswith(RASTERS_PREFIX) and not m.endswith("/")]
        for i, member in enumerate(members, 1):
            # Flatten: strip RASTERS_PREFIX -> "<category>/<file>"
            rel = member[len(RASTERS_PREFIX):]
            target = rasters_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if i % 25 == 0 or i == len(members):
                print(f"\r    {i}/{len(members)} files extracted ...", end="", flush=True)
    print(f"\n  Rasters ready at: {rasters_root}")

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"  Removed archive: {zip_path.name}")


def download_models(dest: Path, keep_zip: bool) -> None:
    """Download MODELS_ZIP and extract the checkpoints into ./models/."""
    models_root = dest / "models"
    if all((models_root / m).exists() for m in EXPECTED_MODELS):
        print(f"Model checkpoints already present in: {models_root}")
        return

    zip_path = dest / MODELS_ZIP
    _download(f"{BASE_URL}/{MODELS_ZIP}", zip_path)

    print("  Extracting model checkpoints ...")
    models_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # Normalise: ensure each checkpoint ends up directly inside ./models/
    for fname in EXPECTED_MODELS:
        if not (models_root / fname).exists():
            found = next(dest.rglob(fname), None)
            if found is not None and found != models_root / fname:
                shutil.move(str(found), str(models_root / fname))

    present = [m for m in EXPECTED_MODELS if (models_root / m).exists()]
    print(f"  Model checkpoints ready in: {models_root}  ({', '.join(present)})")

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"  Removed archive: {zip_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Download the AERO-DETR dataset + models from Zenodo ({ZENODO_DOI}).")
    parser.add_argument("--dest", type=str, default=".",
                        help="Destination directory (default: current directory).")
    parser.add_argument("--skip-rasters", action="store_true",
                        help="Do not download/extract the GIS rasters.")
    parser.add_argument("--skip-models", action="store_true",
                        help="Do not download/extract the model checkpoints.")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the downloaded .zip archives after extraction.")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"AERO-DETR dataset download  |  Zenodo {ZENODO_DOI}")
    print(f"Destination: {dest}\n")

    if not args.skip_rasters:
        print("[1/2] GIS rasters")
        download_rasters(dest, args.keep_zip)
    else:
        print("[1/2] GIS rasters — skipped")

    if not args.skip_models:
        print("\n[2/2] Model checkpoints")
        download_models(dest, args.keep_zip)
    else:
        print("\n[2/2] Model checkpoints — skipped")

    print("\nDone. Expected layout:")
    print("  rasters/<category>/*.jp2")
    print("  models/*.pt")
    print("  ground_truth/*.geojson   (shipped in this repo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
