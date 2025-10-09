"""Delete the .training directory recursively. Use with caution.

Run:
  python tools/clean_training_dir.py
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / ".training"

if TRAINING.exists():
    print(f"Removing {TRAINING} ...")
    shutil.rmtree(TRAINING)
    print("Removed.")
else:
    print(f"No {TRAINING} directory found.")
