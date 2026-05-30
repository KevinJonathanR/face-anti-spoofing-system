import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


def fast_image_hash(img: Image.Image) -> str:
    return hashlib.md5(img.resize((64, 64)).convert("L").tobytes()).hexdigest()


def prepare_and_clean_dataset(json_file: str, input_folder: str, clean_folder: str) -> None:
    """
    Copy the training dataset and relocate mislabeled images based on a JSON manifest.

    The JSON file contains a list of dicts with keys:
        - file_name: image filename
        - source_folder: correct label folder
        - target_folder: where the image was incorrectly placed
    """
    input_dir = Path(input_folder)
    clean_dir = Path(clean_folder)
    json_path = Path(json_file)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON manifest not found: {json_path}")

    if not clean_dir.exists():
        shutil.copytree(input_dir, clean_dir)
        print("Dataset copied.")

    with open(json_path) as f:
        misplaced = json.load(f)

    moved, failed = 0, 0
    for item in misplaced:
        src = clean_dir / item["target_folder"] / item["file_name"]
        dst = clean_dir / item["source_folder"] / item["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.exists():
                shutil.move(str(src), str(dst))
                moved += 1
            else:
                print(f"  Not found: {src.name}")
                failed += 1
        except Exception as e:
            print(f"  Error moving {item['file_name']}: {e}")
            failed += 1

    print(f"Cleaning done. Moved: {moved}, Failed: {failed}")


def deduplicate(dataset, desc: str = "Dedup") -> tuple:
    """Remove exact duplicate images using MD5 hash on a 64×64 grayscale thumbnail."""
    seen = set()
    keep = []
    for idx in tqdm(range(len(dataset)), desc=desc):
        h = fast_image_hash(dataset[idx]["image"])
        if h not in seen:
            seen.add(h)
            keep.append(idx)
    n_removed = len(dataset) - len(keep)
    return dataset.select(keep), n_removed
