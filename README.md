# AERO-DETR — ICPR 2026 Reproducibility Companion

Reproduces **Table 3** (validation metrics per runway configuration) from the AERO-DETR
runway-extraction pipeline: coarse oriented-bounding-box detection (YOLOv8-OBB) →
horizontal normalisation → marking/runway detection (RT-DETR) → polygon reconstruction →
per-airport IoU vs. ground truth.

## One-click quick check (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spatiallysaying/AERO-DETR-ICPR2026-Companion/blob/main/Reproduce_AERO_DETR_Pipeline.ipynb)

Click the badge to open `Reproduce_AERO_DETR_Pipeline.ipynb` directly in Colab, then
**Runtime → Run all** (select a **GPU** runtime first). The notebook clones the repo,
installs dependencies, downloads the Zenodo dataset + checkpoints, runs the full pipeline,
and rebuilds Table 3 — no local setup required.

- **Dataset & model checkpoints:** Zenodo — DOI [10.5281/zenodo.21094439](https://doi.org/10.5281/zenodo.21094439)
- **Imagery format:** georeferenced JPEG2000 (`*.jp2`, *lossy* compression). The archived
  rasters are lossy-compressed, so per-category IoU may deviate by up to ~1 point from a run
  on the original lossless GeoTIFFs; the aggregate IoU is unaffected.
- Large artifacts (`rasters/`, `models/`) are **not** committed — they are downloaded from
  Zenodo by `1_download_dataset.py`. Ground-truth vectors ship inside this repo.

---

## Repository layout

```
1_download_dataset.py     # fetch rasters/*.jp2 + models/*.pt from Zenodo
2_run_inference.py        # headless full pipeline -> metrics/reproduce_table3.csv
3_ablation_fixes.py       # ablation A: fallback + 1:1 matching (offline, no GPU)
4_ablation_hullmar.py     # ablation B: hull->MAR + padding sweep -> metrics/ablation_hullmar.csv
5_make_hull_mar_fig.py    # renders figures/hull_vs_mar.png
Reproduce_AERO_DETR_Pipeline.ipynb   # interactive notebook: pipeline + visuals + Table 3
requirements.txt
geo_utils.py, extract_runway_markings.py, geopyseg/   # pipeline modules
ground_truth/             # *.geojson ground-truth runways (shipped)
metrics/                  # *.csv metric outputs (Table 3, timings, ablations)
rasters/     (gitignored) # <category>/*.jp2  — downloaded from Zenodo
models/      (gitignored) # *.pt checkpoints  — downloaded from Zenodo
```

Runway configurations (categories): `single`, `parallel`, `inter`, `mixed`, `complex`.

---

## Quick start (local, GPU recommended)

```bash
git clone https://github.com/spatiallysaying/AERO-DETR-ICPR2026-Companion.git
cd AERO-DETR-ICPR2026-Companion

# 1. Install dependencies (see the Colab section for GDAL notes)
pip install -r requirements.txt

# 2. Download imagery (~5 GB) + model checkpoints (~253 MB) from Zenodo
python 1_download_dataset.py

# 3. Reproduce Table 3 (headless; all five categories)
python 2_run_inference.py

# ...or run a single category
python 2_run_inference.py --categories inter
```

Outputs land in `metrics/` (`reproduce_table3.csv`, `reproduce_output_times.csv`).
Per-airport intermediate predictions are written to `reproduce_output_<category>/`.

### Interactive notebook

Open `Reproduce_AERO_DETR_Pipeline.ipynb` and run top-to-bottom. The first setup cell
installs dependencies and downloads the dataset/models; later cells display the input
imagery, overlay predictions vs. ground truth, and rebuild Table 3.

### Ablations (optional, offline — reuse step-2 outputs, no GPU)

```bash
python 3_ablation_fixes.py       # fallback + 1:1 matching effect on IoU
python 4_ablation_hullmar.py     # hull->MAR + padding sweep -> metrics/ablation_hullmar.csv
python 5_make_hull_mar_fig.py    # -> figures/hull_vs_mar.png
```

---

## Google Colab

GDAL needs system libraries before the Python wheel. In a fresh Colab notebook:

```python
!git clone https://github.com/spatiallysaying/AERO-DETR-ICPR2026-Companion.git
%cd AERO-DETR-ICPR2026-Companion

# GDAL system libs, then a matching wheel
!apt-get -qq install -y gdal-bin libgdal-dev
!pip install -q GDAL==$(gdal-config --version)

# Remaining Python deps
!pip install -q -r requirements.txt

# Download dataset (rasters/*.jp2) + models (*.pt) from Zenodo (~5 GB, one time)
!python 1_download_dataset.py

# Reproduce Table 3 (pick a category to keep runtime modest)
!python 2_run_inference.py --categories inter
```

Use a GPU runtime (**Runtime → Change runtime type → GPU**) for reasonable inference times.
The `Reproduce_AERO_DETR_Pipeline.ipynb` setup cell performs the same steps automatically.

---

## `1_download_dataset.py` options

```
python 1_download_dataset.py                 # rasters + models into ./
python 1_download_dataset.py --skip-models   # imagery only
python 1_download_dataset.py --skip-rasters  # checkpoints only
python 1_download_dataset.py --keep-zip      # keep the .zip after extracting
```

It is idempotent — existing `rasters/*.jp2` and `models/*.pt` are left untouched.

---

## Metrics

All metric summaries are written to `metrics/`:

| File | Produced by | Contents |
|------|-------------|----------|
| `reproduce_table3.csv` | `2_run_inference.py` | Table 3 (IoU mean, airports ≥ 0.8 IoU, timing) per category |
| `reproduce_output_times.csv` | `2_run_inference.py` | per-airport inference time |
| `reproduce_output_table3.csv` | notebook | Table 3 from the notebook run |
| `reproduce_output_all_times.csv` | notebook | per-raster timing (notebook) |
| `reproduce_seed_variability.csv` | notebook | multi-seed IoU mean ± std |
| `ablation_hullmar.csv` | `4_ablation_hullmar.py` | hull vs. MAR + padding IoU sweep |

---

## Citation

If you use this pipeline or dataset, please cite the AERO-DETR paper and the Zenodo
record ([10.5281/zenodo.21094439](https://doi.org/10.5281/zenodo.21094439)).
