"""Compatibility entrypoint for the AERO-DETR reproduction CLI.

Canonical execution follows numbered scripts:
    1) 1_download_dataset.py
    2) 2_run_inference.py

This file exists only because some documents/reviews refer to `run_inference.py`.
It forwards to the canonical `2_run_inference.py` script.
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).with_name("2_run_inference.py")
    runpy.run_path(str(target), run_name="__main__")
