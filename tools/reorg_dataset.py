"""
Reorganize a local dataset (folder or zip) into the project's .training/<label>/ folders.

Usage examples:
  # Dry-run: show what would be moved
  python tools/reorg_dataset.py --source path/to/extracted_dataset --mapping mapping.json --dry-run

  # Move files using simple filename-based label inference
  python tools/reorg_dataset.py --source path/to/extracted_dataset --infer-labels

  # Unpack a zip and reorganize
  python tools/reorg_dataset.py --source path/to/archive.zip --mapping mapping.json

mapping.json format (optional): a JSON object mapping source filenames or source labels to target labels.
Example:
  {
    "cardboard": "paper",
    "plastic_bottle": "plastic",
    "class1.jpg": "plastic"
  }

The script will create target folders under .training/<label>/ if they don't exist and copy files safely (no overwrite unless --force).

"""
from __future__ import annotations
import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / ".training"
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def load_mapping(path: Optional[Path] = None, mapping_str: Optional[str] = None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    elif mapping_str:
        mapping = json.loads(mapping_str)
    return mapping


def find_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            yield p


def safe_copy(src: Path, dest: Path, force: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        # avoid overwriting; create a unique name
        base = dest.stem
        ext = dest.suffix
        for i in range(1, 1000):
            candidate = dest.with_name(f"{base}_{i}{ext}")
            if not candidate.exists():
                shutil.copy2(src, candidate)
                return candidate
        raise FileExistsError(f"Too many collisions copying {dest}")
    else:
        shutil.copy2(src, dest)
        return dest


def unpack_if_zip(source: Path, temp_dir: Path) -> Path:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as z:
            z.extractall(temp_dir)
        return temp_dir
    return source


def infer_label_from_path(p: Path) -> Optional[str]:
    # heuristics: look in parent folder names for a label-like token
    for part in reversed(p.parts):
        name = part.lower()
        # ignore common names
        if name in {"images", "img", "dataset", "train", "val", "test", "jpg", "png"}:
            continue
        # short-circuit on single-token label
        if len(name) < 64 and all(c.isalnum() or c in "-_" for c in name):
            return name
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description="Reorganize dataset into .training/<label>/")
    p.add_argument("--source", required=True, help="Path to folder or zip containing images")
    p.add_argument("--mapping", help="Path to JSON mapping file mapping source labels/filenames to target labels")
    p.add_argument("--mapping-str", help="Inline JSON mapping string")
    p.add_argument("--dry-run", action="store_true", help="Don't copy, only show actions")
    p.add_argument("--force", action="store_true", help="Allow overwriting existing files")
    p.add_argument("--infer-labels", action="store_true", help="Infer labels from folder names when mapping not provided")
    args = p.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"Source not found: {source}")
        return 2

    mapping = load_mapping(Path(args.mapping) if args.mapping else None, args.mapping_str)

    temp_unpack = None
    try:
        if source.is_file() and source.suffix.lower() == ".zip":
            temp_unpack = Path(".tmp_reorg_unzip")
            if temp_unpack.exists():
                shutil.rmtree(temp_unpack)
            temp_unpack.mkdir()
            work_root = unpack_if_zip(source, temp_unpack)
        else:
            work_root = source

        actions = []
        for img in find_images(work_root):
            rel = img.name
            target_label = None
            # mapping by filename exact match
            if rel in mapping:
                target_label = mapping[rel]
            else:
                # mapping by parent folder name
                parent_key = img.parent.name
                if parent_key in mapping:
                    target_label = mapping[parent_key]
                elif args.infer_labels:
                    inferred = infer_label_from_path(img.relative_to(work_root))
                    if inferred:
                        target_label = inferred

            if not target_label:
                # fallback to placing under 'general'
                target_label = "general"

            dest = TRAINING_DIR / target_label / img.name
            actions.append((img, dest))

        if args.dry_run:
            print("Dry run - actions:")
            for s, d in actions:
                print(f"{s} -> {d}")
            return 0

        # perform copies
        for s, d in actions:
            print(f"Copying {s} -> {d}")
            safe_copy(s, d, force=args.force)

        print("Done. Copied", len(actions), "files.")
        return 0
    finally:
        if temp_unpack and temp_unpack.exists():
            shutil.rmtree(temp_unpack)


if __name__ == "__main__":
    raise SystemExit(main())
