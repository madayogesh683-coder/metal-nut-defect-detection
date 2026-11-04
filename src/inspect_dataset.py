
from pathlib import Path
import argparse
import sys

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

DEFECT_TYPES = ["bent", "color", "flip", "scratch"]

def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

def inspect_dataset(dataset_path):
    root = Path(dataset_path)

    required = [
        root / "train" / "good",
        root / "test" / "good",
    ]

    for defect in DEFECT_TYPES:
        required.append(root / "test" / defect)
        required.append(root / "ground_truth" / defect)

    missing = [p for p in required if not p.is_dir()]

    print("\n========== DATASET INSPECTION ==========")

    if missing:
        print("\nMissing folders:")
        for folder in missing:
            print(folder)
        return False

    print("\nNormal Images:")

    for split in ["train", "test"]:
        folder = root / split / "good"
        print(f"{split}/good: {count_images(folder)}")

    total = 0

    print("\nDefective Images:")

    for defect in DEFECT_TYPES:
        image_folder = root / "test" / defect
        mask_folder = root / "ground_truth" / defect

        images = [
            p for p in image_folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        masks = [
            p for p in mask_folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        mask_names = {p.name.lower() for p in masks}

        missing_masks = [
            p.name for p in images
            if f"{p.stem}_mask{p.suffix}".lower()
            not in mask_names
        ]

        print(f"\n{defect}:")
        print("Images:", len(images))
        print("Masks:", len(masks))
        print("Missing masks:", len(missing_masks))

        total += len(images)

    print("\nTotal defective images:", total)
    print("\nInspection completed successfully!")

    return total > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="dataset/metal_nut"
    )

    args = parser.parse_args()

    success = inspect_dataset(args.dataset)

    sys.exit(0 if success else 1)
    